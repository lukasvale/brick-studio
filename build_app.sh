#!/bin/bash
# Compiles the native parts and assembles "Brick Studio.app" in this folder.
# The app runs the Python engine next to it, so keep the app inside this folder.
set -euo pipefail
cd "$(dirname "$0")"

command -v swiftc >/dev/null || { echo "swiftc not found. Install Xcode Command Line Tools: xcode-select --install"; exit 1; }

echo "Building the Vision foreground guide…"
swiftc -O native/ForegroundGuide.swift -o native/foreground-guide -framework Vision -framework CoreImage

echo "Building Brick Studio…"
APP="Brick Studio.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp app/Info.plist "$APP/Contents/Info.plist"
cp app/BrickStudio.icns "$APP/Contents/Resources/BrickStudio.icns"
swiftc -O -parse-as-library native/BrickStudio.swift -o "$APP/Contents/MacOS/BrickStudio" -framework SwiftUI -framework AppKit

# Ad-hoc signature so macOS will launch a locally built app.
codesign --force --sign - "$APP" >/dev/null
echo "Done: $(pwd)/$APP"
