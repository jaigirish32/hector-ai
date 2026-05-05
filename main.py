import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from paths import resource_path
from theme import apply_theme
from windows.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("HECTOR-AI")
    app.setApplicationDisplayName("HECTOR-AI")
    app.setOrganizationName("Karri")

    # Icon is optional — silently skip if the asset isn't present (dev
    # checkouts where assets/ hasn't been added, or stripped builds).
    icon_path = resource_path("assets/logo.jpeg")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    apply_theme(app)

    window = MainWindow()
    window.show()

    # Belt-and-suspenders graceful shutdown. MainWindow.closeEvent
    # handles the normal close-button path. aboutToQuit catches any
    # other exit path (Ctrl+C, OS-forced quit, programmatic exit).
    # Both call ComparisonView.shutdown() which delegates to
    # Dispatcher.shutdown(); the dispatcher's shutdown is idempotent.
    app.aboutToQuit.connect(window.comparison_view.shutdown)

    exit_code = app.exec()

    # Force-exit. Workers stuck in retry-sleep or inside a blocking SDK
    # call do not observe the cancel flag set by Dispatcher.shutdown(),
    # so they remain alive in the QThreadPool after app.exec() returns.
    # Python's normal exit waits for non-daemon threads, including those
    # workers — which can keep the process (and the launching terminal)
    # alive indefinitely.
    #
    # By the time we reach here, the user has already chosen to close
    # the app. Force-exit via os._exit is the standard pattern for Qt
    # GUI apps in this situation: it skips Python's normal teardown
    # (which is what's hanging) and immediately terminates the process.
    # OS reaps sockets, kills threads, releases handles. Anthropic and
    # other providers handle abrupt client disconnect — that's normal
    # HTTP — so no real-world data loss.
    #
    # Step 7's retry helper rework will make in-flight cancellation
    # responsive to cancel_flag (currently the helper uses time.sleep()
    # which ignores the flag). Once that lands, more workers will exit
    # cleanly within Dispatcher.shutdown's wait window. But os._exit()
    # remains the right safety net for shutdown — it ensures the
    # process always exits, regardless of any worker that didn't comply.
    os._exit(exit_code)


if __name__ == "__main__":
    main()