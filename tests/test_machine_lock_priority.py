"""Голосовое вне очереди: пакет пропускает его вперёд и отдаёт машину между окнами.

2026-10-10 диктовка Handy ждала 1517 с, пока пакет подкастов считал один
выпуск. Здесь — без моделей и звука: «пакет», «голосовое» и «воркер» — это
отдельные процессы, которые берут тот же замок во временном каталоге и пишут
в общий журнал, кто когда работал. Ждём событий, а не времени: под нагрузкой
(load average 60, 2026-10-04) таймеры в тестах замка уже ложно падали.

Классы unittest: pytest их собирает вместе с остальными, а `python -m
unittest` гоняет файл и без conftest.
"""
from __future__ import annotations

import argparse
import contextlib
import errno
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "skills" / "transcriber" / "scripts"
HANDY = ROOT / "integrations" / "handy"
# conftest кладёт пакет в путь только для pytest; unittest его не читает.
for _path in (SCRIPTS, HANDY):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from audio_transcription import machine_lock  # noqa: E402
from audio_transcription.machine_lock import PRIORITY_NAME, PRIORITY_STALE_SECONDS, machine_slot  # noqa: E402

PRELUDE = """
import json, os, subprocess, sys, time
from pathlib import Path
from audio_transcription import isolated, machine_lock
from audio_transcription.machine_lock import machine_slot
lock, log = Path(sys.argv[1]), Path(sys.argv[2])
marker = lock.with_name("priority.wait")
def note(text):
    with log.open("a", encoding="utf-8") as handle:
        handle.write(text + "\\n")
"""

VOICE = PRELUDE + """
with machine_slot("handy-голосовое", path=lock, priority=True, poll_seconds=0.02) as slot:
    note("voice-start")
    time.sleep(float(sys.argv[3]))
    note(f"voice-end {slot.waited:.3f}")
"""

# Пакет из двух записей без зазора между ними: голосовое не успело бы
# вклиниться, если бы пакет не пропускал его сам.
BATCH_TWO_FILES = PRELUDE + """
with machine_slot("пакет-1", path=lock, poll_seconds=0.02):
    note("held")
    deadline = time.monotonic() + 60
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
with machine_slot("пакет-2", path=lock, poll_seconds=0.02):
    note("batch-2")
"""

BATCH_GIVES_WAY = PRELUDE + """
with machine_slot("пакет", path=lock, poll_seconds=0.02) as slot:
    note("held")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if machine_lock.priority_waiting():
            note("yield")
            machine_lock.give_way()
            note("resume")
            break
        time.sleep(0.01)
    note(f"yielded {slot.yielded:.3f}")
"""

# Воркер считает «окна», пока не появится stop; после каждого окна — точка
# паузы, как у настоящего воркера в worker.py.
FAKE_WORKER = """
import json, sys, time
from pathlib import Path
from audio_transcription.isolated import hold_while_paused
progress, log, stop = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
window = 0
while not stop.exists() and window < 5000:
    time.sleep(0.03)
    window += 1
    progress.write_text(json.dumps({"completed": window, "total": 0}), encoding="utf-8")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"window {window}\\n")
    hold_while_paused(progress, poll_seconds=0.02)
"""

BATCH_WITH_WORKER = PRELUDE + """
job, stop = Path(sys.argv[3]), Path(sys.argv[4])
progress = job / "progress.json"
with machine_slot("пакет", path=lock, poll_seconds=0.02) as slot:
    note("held")
    worker = subprocess.Popen([sys.executable, "-c", sys.argv[5], str(progress), str(log), str(stop)])
    # Голосовое держит машину дольше, чем терпит watchdog: простой на
    # голосовом не должен считаться зависанием воркера.
    done = isolated.wait_for_progress(worker, progress, stall_timeout_seconds=1.0, poll_seconds=0.02)
    note(f"done {done} {worker.returncode} {slot.yielded:.3f}")
"""


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(SCRIPTS), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return environment


def _wait_for(predicate, *, timeout: float = 60.0, what: str = "событие") -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"не дождались: {what}")
        time.sleep(0.01)


class _Folder(unittest.TestCase):
    """Временный каталог вместо ~/.transcriber и журнал событий процессов."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.lock = self.dir / "run.lock"
        self.marker = self.dir / PRIORITY_NAME
        self.log = self.dir / "events.log"
        self.processes: list[subprocess.Popen] = []

    def tearDown(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
        self._tmp.cleanup()

    def spawn(self, code: str, *args: str) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(self.lock), str(self.log), *args],
            env=_environment(),
        )
        self.processes.append(process)
        return process

    def events(self) -> list[str]:
        try:
            return self.log.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []

    def logged(self, prefix: str) -> bool:
        return any(line.startswith(prefix) for line in self.events())

    def finish(self, *processes: subprocess.Popen) -> None:
        for process in processes:
            self.assertEqual(process.wait(timeout=60), 0)

    def index(self, prefix: str) -> int:
        return next(number for number, line in enumerate(self.events()) if line.startswith(prefix))

    def write_marker(self, pid: int, label: str = "handy-1", age_seconds: float = 0.0) -> None:
        self.marker.write_text(json.dumps({"pid": pid, "label": label}), encoding="utf-8")
        if age_seconds:
            moment = time.time() - age_seconds
            os.utime(self.marker, (moment, moment))


class VoiceJumpsTheQueueTest(_Folder):
    def test_batch_lets_voice_go_before_its_next_file(self) -> None:
        batch = self.spawn(BATCH_TWO_FILES)
        _wait_for(lambda: self.logged("held"), what="пакет взял машину")
        voice = self.spawn(VOICE, "0.3")
        self.finish(batch, voice)

        names = [line.split()[0] for line in self.events()]
        self.assertEqual(names, ["held", "voice-start", "voice-end", "batch-2"])
        self.assertGreater(float(self.events()[2].split()[1]), 0)
        self.assertFalse(self.marker.exists(), "взявший машину снимает метку")

    def test_holder_gives_way_and_takes_the_machine_back(self) -> None:
        batch = self.spawn(BATCH_GIVES_WAY)
        _wait_for(lambda: self.logged("held"), what="пакет взял машину")
        voice = self.spawn(VOICE, "0.3")
        self.finish(batch, voice)

        names = [line.split()[0] for line in self.events()]
        self.assertEqual(names, ["held", "yield", "voice-start", "voice-end", "resume", "yielded"])
        self.assertGreater(float(self.events()[-1].split()[1]), 0)
        self.assertFalse(self.marker.exists())

    def test_worker_pauses_between_windows_while_voice_runs(self) -> None:
        job = self.dir / "job"
        job.mkdir()
        stop = self.dir / "stop"
        batch = self.spawn(BATCH_WITH_WORKER, str(job), str(stop), FAKE_WORKER)
        _wait_for(lambda: self.logged("window 3"), what="воркер считает окна")
        voice = self.spawn(VOICE, "1.5")
        self.finish(voice)
        _wait_for(lambda: any(line.startswith("window") for line in self.events()[self.index("voice-end"):]),
                  what="воркер продолжил после голосового")
        stop.touch()
        self.finish(batch)

        events = self.events()
        start, end = self.index("voice-start"), self.index("voice-end")
        during = [line for line in events[start:end] if line.startswith("window")]
        # Конца окна родитель не ждёт: одно начатое окно может закончиться
        # уже при голосовом, следующее — нет.
        self.assertLessEqual(len(during), 1, events)
        self.assertTrue(any(line.startswith("window") for line in events[end:]))
        done = events[-1].split()
        self.assertEqual(done[:3], ["done", "True", "0"], "watchdog не убил воркер на простое")
        self.assertGreater(float(done[3]), 0)
        self.assertFalse((job / "pause").exists())


class PriorityMarkerTest(_Folder):
    def test_free_machine_is_taken_without_a_marker(self) -> None:
        calls: list[str] = []
        with machine_slot("голосовое", path=self.lock, priority=True,
                          on_wait=lambda *_: calls.append("wait")) as slot:
            self.assertFalse(self.marker.exists())
        self.assertEqual(calls, [])
        self.assertIsInstance(slot.waited, float)
        self.assertEqual(slot.yielded, 0.0)
        self.assertEqual(sorted(path.name for path in self.dir.iterdir()), ["run.lock"])

    def test_waiting_voice_marks_atomically_and_unmarks_when_interrupted(self) -> None:
        seen: list[dict] = []

        class Interrupted(Exception):
            pass

        def interrupt(holder: str, waited: float) -> None:
            seen.append(json.loads(self.marker.read_text(encoding="utf-8")))
            raise Interrupted

        with machine_slot("пакет", path=self.lock):
            with self.assertRaises(Interrupted):
                with machine_slot("handy-2", path=self.lock, priority=True, on_wait=interrupt):
                    self.fail("машина занята — сюда не попасть")
            self.assertEqual(seen, [{"pid": os.getpid(), "label": "handy-2"}])
            self.assertFalse(self.marker.exists(), "метка снимается и при обрыве ожидания")
            # Временный файл атомарной записи переименован, а не брошен рядом.
            self.assertEqual(sorted(path.name for path in self.dir.iterdir()), ["run.lock"])

    def test_fresh_marker_holds_back_a_free_machine(self) -> None:
        self.write_marker(os.getpid())
        holders: list[str] = []
        entered = threading.Event()

        def batch() -> None:
            with machine_slot("пакет", path=self.lock, poll_seconds=0.01,
                              on_wait=lambda holder, waited: holders.append(holder)):
                entered.set()

        worker = threading.Thread(target=batch)
        worker.start()
        _wait_for(lambda: holders, what="пакет увидел метку")
        self.assertFalse(entered.is_set(), "свободную машину пакет не берёт, пока ждёт голосовое")
        self.marker.unlink()
        worker.join(timeout=10)
        self.assertTrue(entered.is_set())
        self.assertIn("голосовое вне очереди: handy-1", holders[0])

    def test_stale_or_orphaned_marker_is_ignored(self) -> None:
        finished = subprocess.Popen([sys.executable, "-c", "pass"])
        finished.wait(timeout=30)
        for pid, age in ((finished.pid, 0.0), (os.getpid(), PRIORITY_STALE_SECONDS + 60)):
            with self.subTest(pid=pid, age=age):
                self.write_marker(pid, age_seconds=age)
                calls: list[str] = []
                entered = threading.Event()

                def batch() -> None:
                    with machine_slot("пакет", path=self.lock, poll_seconds=0.01,
                                      on_wait=lambda *_: calls.append("wait")):
                        entered.set()

                # В потоке, чтобы поломка давала провал, а не вечное ожидание.
                threading.Thread(target=batch, daemon=True).start()
                self.assertTrue(entered.wait(timeout=10), "брошенная метка держит машину")
                self.assertEqual(calls, [])

    def test_priority_holder_never_gives_way(self) -> None:
        with machine_slot("голосовое", path=self.lock, priority=True) as slot:
            self.write_marker(os.getpid(), label="handy-3")
            self.assertFalse(slot.wanted())
            self.assertFalse(machine_lock.priority_waiting())
            self.assertEqual(machine_lock.give_way(), 0.0)
        self.assertFalse(machine_lock.priority_waiting())
        self.assertEqual(machine_lock.give_way(), 0.0, "вне прогона уступать нечего")


class WatchdogTest(_Folder):
    def test_time_given_to_voice_is_not_a_stall(self) -> None:
        from audio_transcription import isolated

        progress = self.dir / "progress.json"
        clock = [0.0]
        asked = [True]
        pause_seen: list[str] = []

        class Process:
            returncode = None
            polls = 0

            def poll(self):
                self.polls += 1
                if self.polls > 5:
                    self.returncode = 0
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.returncode = -9

            def wait(self, timeout=None):
                return self.returncode

        def waiting() -> bool:
            # Голосовое ждёт один раз; прогресс за всё время не меняется.
            answer, asked[0] = asked[0], False
            return answer

        def give_way() -> float:
            pause_seen.append((self.dir / "pause").read_text(encoding="utf-8"))
            clock[0] += 10.0  # голосовое считалось дольше, чем терпит watchdog
            return 10.0

        with mock.patch.object(isolated.time, "monotonic", lambda: clock[0]), \
                mock.patch.object(isolated.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
                mock.patch.object(isolated.machine_lock, "priority_waiting", waiting), \
                mock.patch.object(isolated.machine_lock, "give_way", give_way):
            done = isolated.wait_for_progress(Process(), progress, stall_timeout_seconds=1.0, poll_seconds=0.1)

        self.assertTrue(done, "простой на голосовом watchdog считает зависанием")
        self.assertEqual(pause_seen, [str(os.getpid())], "воркер стоит, пока машина у голосового")
        self.assertFalse((self.dir / "pause").exists())


def _denied(real_open, path, mode, *args, **kwargs):
    raise PermissionError(errno.EACCES, "Permission denied", str(path))


def _disk_full(real_open, path, mode, *args, **kwargs):
    """Как на полном диске: файл создан, а запись в него падает."""
    handle = real_open(path, mode, *args, **kwargs)

    class Full:
        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            handle.close()

        def write(self, _data):
            raise OSError(errno.ENOSPC, "No space left on device")

        def close(self) -> None:
            handle.close()

    return Full()


class PauseMarkFailureTest(_Folder):
    """Метку паузы записать не вышло: машину не отдаём, watchdog работает.

    Ревью `--priority`: при сбое записи родитель крутился вхолостую с
    обнулённым на каждом шаге отсчётом watchdog, а полный диск оставлял
    пустой `pause`, на котором воркер стоял, хотя машину никто не отдавал.
    """

    def wait_with_broken_pause(self, write) -> tuple[bool, list[str], str]:
        from audio_transcription import isolated

        clock = [0.0]
        given: list[str] = []

        class Process:
            returncode = None
            polls = 0

            def poll(self):
                # Предохранитель: холостой цикл кончается «успехом», который
                # тест и ловит, а не висит вечно.
                self.polls += 1
                if self.returncode is None and self.polls > 1000:
                    self.returncode = 0
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.returncode = -9

            def wait(self, timeout=None):
                return self.returncode

        real_open = Path.open

        def broken_open(path, mode="r", *args, **kwargs):
            if "w" not in mode:
                return real_open(path, mode, *args, **kwargs)
            return write(real_open, path, mode, *args, **kwargs)

        def give_way() -> float:
            given.append("give_way")
            return 0.0

        errors = io.StringIO()
        # Голосовое ждёт всё время, прогресса нет: воркер и правда завис.
        with mock.patch.object(isolated.time, "monotonic", lambda: clock[0]), \
                mock.patch.object(isolated.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)), \
                mock.patch.object(isolated.machine_lock, "priority_waiting", lambda: True), \
                mock.patch.object(isolated.machine_lock, "give_way", give_way), \
                mock.patch.object(Path, "open", broken_open), \
                contextlib.redirect_stderr(errors):
            done = isolated.wait_for_progress(Process(), self.dir / "progress.json",
                                              stall_timeout_seconds=1.0, poll_seconds=0.1)
        return done, given, errors.getvalue()

    def check(self, write) -> None:
        done, given, errors = self.wait_with_broken_pause(write)
        self.assertFalse(done, "watchdog выключен, пока голосовое ждёт, а метки нет")
        self.assertEqual(given, [], "воркер не встал — машину отдавать нельзя")
        self.assertEqual(errors.count("Внимание"), 1, errors)
        self.assertEqual(sorted(path.name for path in self.dir.iterdir()), [],
                         "ни метки паузы, ни временного файла")

    def test_unwritable_pause_keeps_the_machine_and_the_watchdog(self) -> None:
        self.check(_denied)

    def test_full_disk_leaves_no_empty_pause(self) -> None:
        self.check(_disk_full)


# Подставной mlx_audio.vad для diarization_worker: Sortformer без весов и MLX.
# Каждый чанк отмечается в журнале рядом со звуком в момент, когда воркер его
# попросил.
SORTFORMER_STUB = """
from pathlib import Path
from types import SimpleNamespace


class Model:
    def generate_stream(self, audio_path, **_options):
        journal = Path(audio_path).with_name("chunks.log")
        for index in range(1, 4):
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(f"chunk-{index}\\n")
            turn = SimpleNamespace(start=float(index - 1), end=float(index), speaker=index % 2)
            yield SimpleNamespace(segments=[turn])


def load(_name):
    return Model()
"""


class WorkerPauseTest(_Folder):
    HOLD = (
        "import sys; from pathlib import Path; "
        "from audio_transcription.isolated import hold_while_paused; "
        "Path(sys.argv[2]).write_text('started'); "
        "hold_while_paused(Path(sys.argv[1]), poll_seconds=0.01); print('resumed')"
    )

    def run_hold(self, owner: int | str) -> subprocess.Popen:
        progress = self.dir / "progress.json"
        (self.dir / "pause").write_text(str(owner), encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, "-c", self.HOLD, str(progress), str(self.dir / "started")],
            env=_environment(), stdout=subprocess.PIPE, text=True,
        )
        self.processes.append(process)
        return process

    def test_worker_waits_until_parent_takes_the_pause_back(self) -> None:
        worker = self.run_hold(os.getpid())
        _wait_for(lambda: (self.dir / "started").exists(), what="воркер дошёл до паузы")
        time.sleep(0.3)
        self.assertIsNone(worker.poll(), "пока метка паузы есть, воркер стоит")
        (self.dir / "pause").unlink()
        output, _ = worker.communicate(timeout=30)
        self.assertEqual(worker.returncode, 0)
        self.assertIn("resumed", output)

    def test_asr_worker_stops_at_its_progress_point(self) -> None:
        # Настоящий worker.py без модели: отметка «0 окон из 0» пишется до
        # выбора бэкенда, и на ней воркер обязан встать.
        job = self.dir / "job"
        job.mkdir()
        (job / "request.json").write_text(json.dumps({"schema": 1, "backend": "нет-такого", "chunks": []}),
                                          encoding="utf-8")
        (job / "pause").write_text(str(os.getpid()), encoding="utf-8")
        worker = subprocess.Popen(
            [sys.executable, "-m", "audio_transcription.worker",
             str(job / "request.json"), str(job / "result.json"), str(job / "progress.json")],
            env=_environment(), stderr=subprocess.PIPE, text=True,
        )
        self.processes.append(worker)
        _wait_for(lambda: (job / "progress.json").exists(), what="воркер отметил прогресс")
        time.sleep(0.3)
        self.assertIsNone(worker.poll(), "на точке паузы воркер стоит")
        (job / "pause").unlink()
        _, errors = worker.communicate(timeout=60)
        self.assertEqual(worker.returncode, 2)
        self.assertIn("Неизвестный ASR-бэкенд", errors)

    def test_sortformer_worker_stops_after_its_chunk(self) -> None:
        # Настоящий diarization_worker.py с подставным mlx_audio: модель
        # отдаёт чанки по одному и отмечает в журнале каждый. После первого
        # воркер обязан встать на точке паузы, а не просить следующий.
        stub = self.dir / "stub" / "mlx_audio"
        stub.mkdir(parents=True)
        (stub / "__init__.py").write_text("", encoding="utf-8")
        (stub / "vad.py").write_text(SORTFORMER_STUB, encoding="utf-8")
        job = self.dir / "job"
        job.mkdir()
        audio = job / "prepared.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\0\0" * 16000)
        (job / "request.json").write_text(json.dumps({
            "backend": "sortformer", "audio_path": str(audio),
            "config": {"model_name": "stub/sortformer", "offline": True},
        }), encoding="utf-8")
        (job / "pause").write_text(str(os.getpid()), encoding="utf-8")
        environment = _environment()
        environment["PYTHONPATH"] = os.pathsep.join([str(stub.parent), environment["PYTHONPATH"]])
        worker = subprocess.Popen(
            [sys.executable, "-m", "audio_transcription.diarization_worker",
             str(job / "request.json"), str(job / "result.json"), str(job / "progress.json")],
            env=environment, stderr=subprocess.PIPE, text=True,
        )
        self.processes.append(worker)
        chunks = job / "chunks.log"

        def at_first_chunk() -> bool:
            try:
                return json.loads((job / "progress.json").read_text(encoding="utf-8")).get("completed") == 1
            except (OSError, ValueError):
                return False

        _wait_for(at_first_chunk, what="воркер отметил первый чанк")
        time.sleep(0.3)
        self.assertIsNone(worker.poll(), "на точке паузы воркер стоит")
        self.assertEqual(chunks.read_text(encoding="utf-8").split(), ["chunk-1"],
                         "следующий чанк воркер просит только после паузы")
        (job / "pause").unlink()
        _, errors = worker.communicate(timeout=60)
        self.assertEqual(worker.returncode, 0, errors)
        self.assertEqual(chunks.read_text(encoding="utf-8").split(), ["chunk-1", "chunk-2", "chunk-3"])
        self.assertEqual(len(json.loads((job / "result.json").read_text(encoding="utf-8"))["turns"]), 3)

    def test_empty_pause_is_not_a_pause(self) -> None:
        # Такую метку оставлял сбой записи на полном диске: родитель машину не
        # отдавал, а воркер стоял на ней до конца ожидания.
        worker = self.run_hold("")
        output, _ = worker.communicate(timeout=30)
        self.assertEqual(worker.returncode, 0)
        self.assertIn("resumed", output)

    def test_orphaned_worker_exits_instead_of_waiting_forever(self) -> None:
        # В метке чужой pid — так видит её воркер, чей родитель умер.
        worker = self.run_hold(1)
        output, _ = worker.communicate(timeout=30)
        self.assertEqual(worker.returncode, 2)
        self.assertNotIn("resumed", output)


class PriorityFlagTest(unittest.TestCase):
    def test_flag_is_for_one_recording_only(self) -> None:
        from audio_transcription import cli

        single = cli.build_parser().parse_args(["a.wav", "--priority"])
        self.assertTrue(single.priority)
        cli._validate(single)
        batch = cli.build_parser().parse_args(["a.wav", "b.wav", "--priority"])
        with self.assertRaisesRegex(ValueError, "--priority"):
            cli._validate(batch)

    def test_handy_dictation_runs_with_priority(self) -> None:
        import handy_watch

        args = handy_watch.parse_args(["--python", "py", "--transcribe", "t.py", "--output-root", "out"])
        with mock.patch.object(handy_watch.subprocess, "Popen") as popen:
            handy_watch.transcribe(Path("handy-1.wav"), Path("out/x"), Path("out/x.txt"), args, None)
        self.assertIn("--priority", popen.call_args.args[0])

    def test_agent_own_run_runs_with_priority(self) -> None:
        # Наблюдатель запись не взял, и handy_result.py считает её сам: агент
        # ждёт этот текст сейчас, как ждут текст диктовки.
        import handy_result

        args = argparse.Namespace(python="py", transcribe="t.py", mode="max", language="ru")
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "2026-10-17-1200-handy-1"
            with mock.patch.object(handy_result.subprocess, "Popen") as popen, \
                    mock.patch.object(handy_result, "write_marker"), \
                    mock.patch.object(handy_result, "run_log", lambda _out: Path(folder) / "run.log"):
                handy_result.start_own_run(Path("handy-1.wav"), out, args)
            popen.call_args.kwargs["stdout"].close()
        self.assertIn("--priority", popen.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
