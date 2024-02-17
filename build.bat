@RD /S /Q build
@RD /S /Q dist

pyinstaller main.py --onefile --noconsole --name "ImageSegmentationTool" --clean

cd dist
zip -r "ImageSegmentationToolWindows.zip" *
