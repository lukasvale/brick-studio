#!/bin/bash
# Double-click this file in Finder to set up Brick Studio. It checks what is
# needed, runs setup.sh in this folder and opens the app when it is ready.
# Running it again later updates the setup and rebuilds the app.
cd "$(dirname "$0")"
[ -t 1 ] && clear
echo "Brick Studio setup"
echo "=================="
echo

finish() {
  echo
  read -r -n 1 -p "Press any key to close this window." _
  echo
  exit "$1"
}

# 1. Apple's command-line developer tools (to build the app).
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Brick Studio needs Apple's command-line developer tools."
  echo "An installation window is opening. Click Install, wait for it to finish,"
  echo "then double-click 'Set up Brick Studio' again."
  xcode-select --install >/dev/null 2>&1
  finish 1
fi

# 2. Python 3.13, from python.org or Homebrew.
PYTHON=""
for candidate in python3.13 \
                 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13 \
                 /opt/homebrew/bin/python3.13 \
                 /usr/local/bin/python3.13; do
  if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$(command -v "$candidate")"; break; fi
done
if [ -z "$PYTHON" ]; then
  echo "Brick Studio needs Python 3.13."
  echo "The download page is opening. Download the macOS installer for the newest 3.13 release,"
  echo "run it, then double-click 'Set up Brick Studio' again."
  open "https://www.python.org/downloads/macos/"
  finish 1
fi

# 3. Install everything and build the app.
echo "This takes a few minutes and downloads about 1.5 GB. Keep this window open."
echo
if PYTHON="$PYTHON" ./setup.sh; then
  echo
  echo "Brick Studio is ready. Opening it now."
  echo "Next time, open 'Brick Studio.app' in this folder. Keep the app in this folder."
  open "Brick Studio.app"
  finish 0
else
  echo
  echo "Setup did not finish. The messages above say what went wrong."
  echo "Fix that and double-click 'Set up Brick Studio' again, or send this window's text to your support contact."
  finish 1
fi
