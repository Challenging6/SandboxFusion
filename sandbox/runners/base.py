# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import os
import subprocess
import time
import traceback
from typing import Dict, List, Optional

import psutil
import structlog
import platform
import resource

from sandbox.configs.run_config import RunConfig
from sandbox.runners.isolation import tmp_cgroup, tmp_netns, tmp_overlayfs
from sandbox.runners.types import CodeRunArgs, CodeRunResult, CommandRunResult, CommandRunStatus
from sandbox.utils.common import set_permissions_recursively
from sandbox.utils.execution import cleanup_process, ensure_bash_integrity, kill_process_tree, try_decode

logger = structlog.stdlib.get_logger()
config = RunConfig.get_instance_sync()

STREAM_READ_CHUNK_BYTES = 64 * 1024
STREAM_CAPTURE_LIMIT_BYTES = int(os.getenv('SANDBOX_STREAM_CAPTURE_LIMIT_BYTES', 256 * 1024))


class _BoundedStreamCapture:

    def __init__(self, limit_bytes: int):
        self.limit_bytes = max(0, limit_bytes)
        self._captured = bytearray()
        self._truncated_bytes = 0

    def append(self, chunk: bytes):
        if self.limit_bytes <= 0:
            self._truncated_bytes += len(chunk)
            return

        remaining = self.limit_bytes - len(self._captured)
        if remaining > 0:
            self._captured.extend(chunk[:remaining])
        self._truncated_bytes += max(0, len(chunk) - max(remaining, 0))

    def render(self) -> str:
        text = try_decode(bytes(self._captured))
        if self._truncated_bytes > 0:
            text += f'\n...[truncated {self._truncated_bytes} bytes]'
        return text


async def _drain_stream(stream: Optional[asyncio.StreamReader], limit_bytes: int) -> str:
    if stream is None:
        return ''

    capture = _BoundedStreamCapture(limit_bytes)
    while True:
        chunk = await stream.read(STREAM_READ_CHUNK_BYTES)
        if not chunk:
            break
        capture.append(chunk)
    return capture.render()


async def _write_stdin(stream: Optional[asyncio.StreamWriter], stdin: Optional[str]):
    if stream is None:
        return

    try:
        if stdin is not None:
            stream.write(stdin.encode())
            await stream.drain()
    except (BrokenPipeError, ConnectionResetError) as e:
        logger.warning(f'failed to fully write stdin: {e}')
    finally:
        try:
            stream.close()
            await stream.wait_closed()
        except Exception as e:
            logger.warning(f'Failed to close stdin: {e}')


def _read_output_file(path: Optional[str], limit_bytes: int) -> str:
    if not path:
        return ''

    output_path = os.fspath(path)
    if not os.path.exists(output_path):
        return ''

    try:
        size = os.path.getsize(output_path)
        with open(output_path, 'rb') as f:
            captured = f.read(max(0, limit_bytes))
    except Exception as e:
        logger.warning(f'failed to read output file {output_path}: {e}')
        return ''

    text = try_decode(captured)
    truncated_bytes = max(0, size - len(captured))
    if truncated_bytes > 0:
        text += f'\n...[truncated {truncated_bytes} bytes]'
    return text


def build_preexec_fn(memory_limit_MB: int = -1,
                     set_uid: Optional[int] = None,
                     cwd: Optional[str] = None,
                     cpu_limit_s: Optional[int] = None):
    preexec_steps = []
    if set_uid:
        if cwd is not None:
            set_permissions_recursively(cwd, 0o777)
        preexec_steps.append(lambda: os.setuid(set_uid))

    if memory_limit_MB > 0:
        def memory_limit_preexec():
            _, hard_memory_limit_AS = resource.getrlimit(resource.RLIMIT_AS)
            _, hard_memory_limit_DATA = resource.getrlimit(resource.RLIMIT_DATA)
            soft_memory_limit = memory_limit_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (soft_memory_limit, hard_memory_limit_AS))
            resource.setrlimit(resource.RLIMIT_DATA, (soft_memory_limit, hard_memory_limit_DATA))
            if platform.uname().system != "Darwin":
                _, hard_memory_limit_STACK = resource.getrlimit(resource.RLIMIT_STACK)
                resource.setrlimit(resource.RLIMIT_STACK, (soft_memory_limit, hard_memory_limit_STACK))

        preexec_steps.insert(0, memory_limit_preexec)

    if cpu_limit_s is not None and cpu_limit_s > 0:
        def cpu_limit_preexec():
            soft_cpu_limit = int(cpu_limit_s)
            _, hard_cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
            # Keep a 1-second hard-limit buffer when possible, matching local evaluator behavior.
            if hard_cpu_limit in (-1, resource.RLIM_INFINITY) or hard_cpu_limit >= soft_cpu_limit + 1:
                new_hard_limit = soft_cpu_limit + 1
            else:
                new_hard_limit = hard_cpu_limit
            resource.setrlimit(resource.RLIMIT_CPU, (soft_cpu_limit, new_hard_limit))

        preexec_steps.insert(0, cpu_limit_preexec)

    return (lambda: [step() for step in preexec_steps]) if preexec_steps else None


async def run_command_bare(command: str | List[str],
                           timeout: float = 10,
                           stdin: Optional[str] = None,
                           cwd: Optional[str] = None,
                           extra_env: Optional[Dict[str, str]] = None,
                           use_exec: bool = False,
                           preexec_fn=None,
                           stdin_path: Optional[str] = None,
                           stdout_path: Optional[str] = None) -> CommandRunResult:
    try:
        if stdin is not None and stdin_path is not None:
            raise ValueError('stdin and stdin_path cannot both be provided')

        logger.debug(f'running command {command}')
        stdin_handle = open(os.fspath(stdin_path), 'rb') if stdin_path is not None else None
        stdout_handle = None
        if stdout_path is not None:
            os.makedirs(os.path.dirname(os.fspath(stdout_path)) or '.', exist_ok=True)
            stdout_handle = open(os.fspath(stdout_path), 'wb')
        try:
            if use_exec:
                p = await asyncio.create_subprocess_exec(*command,
                                                         stdin=stdin_handle or subprocess.PIPE,
                                                         stdout=stdout_handle or subprocess.PIPE,
                                                         stderr=subprocess.PIPE,
                                                         cwd=cwd,
                                                         env={
                                                             **os.environ,
                                                             **(extra_env or {})
                                                         },
                                                         preexec_fn=preexec_fn)
            else:
                p = await asyncio.create_subprocess_shell(command,
                                                          stdin=stdin_handle or subprocess.PIPE,
                                                          stdout=stdout_handle or subprocess.PIPE,
                                                          stderr=subprocess.PIPE,
                                                          cwd=cwd,
                                                          executable='/bin/bash',
                                                          env={
                                                              **os.environ,
                                                              **(extra_env or {})
                                                          },
                                                          preexec_fn=preexec_fn)
        finally:
            if stdin_handle is not None:
                stdin_handle.close()
            if stdout_handle is not None:
                stdout_handle.close()
        stdout_task = asyncio.create_task(_drain_stream(p.stdout, STREAM_CAPTURE_LIMIT_BYTES))
        stderr_task = asyncio.create_task(_drain_stream(p.stderr, STREAM_CAPTURE_LIMIT_BYTES))
        await _write_stdin(p.stdin, stdin)

        start_time = time.time()
        timed_out = False
        try:
            await asyncio.wait_for(p.wait(), timeout=timeout)
            execution_time = time.time() - start_time
            logger.debug(f'stop running command {command}')
        except asyncio.TimeoutError:
            timed_out = True
            execution_time = time.time() - start_time
        finally:
            if p.returncode is None and psutil.pid_exists(p.pid):
                kill_process_tree(p.pid)
                logger.info(f'process killed: {p.pid}')
                try:
                    await asyncio.wait_for(p.wait(), timeout=1)
                except asyncio.TimeoutError:
                    logger.warning(f'process did not exit promptly after kill: {p.pid}')
            if config.sandbox.cleanup_process:
                cleanup_process()
            if config.sandbox.restore_bash:
                ensure_bash_integrity()

        captured_stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        stdout = _read_output_file(stdout_path, STREAM_CAPTURE_LIMIT_BYTES) if stdout_path else captured_stdout

        if timed_out:
            return CommandRunResult(status=CommandRunStatus.TimeLimitExceeded,
                                    execution_time=execution_time,
                                    stdout=stdout,
                                    stderr=stderr)

        return CommandRunResult(status=CommandRunStatus.Finished,
                                execution_time=execution_time,
                                return_code=p.returncode,
                                stdout=stdout,
                                stderr=stderr)
    except Exception as e:
        message = f'exception on running command {command}: {e} | {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        return CommandRunResult(status=CommandRunStatus.Error, stderr=message)


async def run_commands(compile_command: Optional[str], run_command: str, cwd: str, extra_env: Optional[Dict[str, str]],
                       args: CodeRunArgs, **kwargs) -> CodeRunResult:
    files = {}
    compile_res = None
    run_res = None

    if config.sandbox.isolation == 'none':
        preexec_fn = build_preexec_fn(args.memory_limit_MB, kwargs.get('set_uid'), cwd)

        if compile_command is not None:
            compile_res = await run_command_bare(compile_command,
                                                 args.compile_timeout,
                                                 None,
                                                 cwd,
                                                 extra_env,
                                                 preexec_fn=preexec_fn)
        if compile_res is None or (compile_res.status == CommandRunStatus.Finished and compile_res.return_code == 0):
            run_res = await run_command_bare(run_command,
                                             args.run_timeout,
                                             args.stdin,
                                             cwd,
                                             extra_env,
                                             preexec_fn=preexec_fn)
        for filename in args.fetch_files:
            fp = os.path.abspath(os.path.join(cwd, filename))
            if os.path.isfile(fp):
                with open(fp, 'rb') as f:
                    content = f.read()
                base64_content = base64.b64encode(content).decode('utf-8')
                files[filename] = base64_content
        return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)

    elif config.sandbox.isolation == 'lite':
        async with tmp_overlayfs() as root, tmp_cgroup(mem_limit='4G', cpu_limit=1) as cgroups, tmp_netns(
                kwargs.get('netns_no_bridge', False)) as netns:
            prefix = []
            for cg in cgroups:
                prefix += ['cgexec', '-g', cg]
            if not kwargs.get('disable_pid_isolation', False):
                prefix += ['unshare', '--pid', '--fork', '--mount-proc']
            prefix += ['ip', 'netns', 'exec', netns]
            prefix += ['chroot', root]

            if compile_command is not None:
                compile_res = await run_command_bare(prefix + ['bash', '-c', f'cd {cwd} && {compile_command}'],
                                                     args.compile_timeout, None, cwd, extra_env, True)
            if compile_res is None or (compile_res.status == CommandRunStatus.Finished and
                                       compile_res.return_code == 0):
                run_res = await run_command_bare(prefix + ['bash', '-c', f'cd {cwd} && {run_command}'],
                                                 args.run_timeout, args.stdin, cwd, extra_env, True)

            for filename in args.fetch_files:
                fp = os.path.join(root, os.path.abspath(os.path.join(cwd, filename))[1:])
                if os.path.isfile(fp):
                    with open(fp, 'rb') as f:
                        content = f.read()
                    base64_content = base64.b64encode(content).decode('utf-8')
                    files[filename] = base64_content
            return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)


def restore_files(dir: str, files: Dict[str, Optional[str]]):
    for filename, content in files.items():
        if not isinstance(content, str):
            continue
        if "IGNORE_THIS_FILE" in filename:
            continue
        filepath = os.path.join(dir, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as file:
            file.write(base64.b64decode(content))
