# OpenCV FRC 2016
## Build instructions

There are build dependencies in the *.7z files.  To build, you must do the following:
* `opencv.7z` contains a directory called `opencv.build`. Expand this directory and put it as a neighboring directory to this repo.
* `ntcore.7z` contains a directory called `ntcore.build`.  Do the same thing.
* Copy `ntcore.build/Release/ntcore.dll` to the Windows SysWOW64 directory
* Copy the DLLS in `opencv.build/x86/vc14/bin` to the Windows SysWOW64 directory

Open `opencv_frc_2016.sln` in Visual Studio.  Make sure the architecture is set to x86.  You should be able to build either the Debug or Release versions.
