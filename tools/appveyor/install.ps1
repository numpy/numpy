# Sample script to install Python and pip under Windows
# Authors: Olivier Grisel, Jonathan Helmus and Kyle Kastner
# License: CC0 1.0 Universal: http://creativecommons.org/publicdomain/zero/1.0/

$SEVEN_ZIP_URL = "http://sourceforge.net/projects/sevenzip/files/7-Zip/9.35/7z935-x64.msi/download"
$SEVEN_ZIP_FILENAME = "7z935-x64.msi"
$PYTHOHN_BASE_URL = "https://www.python.org/ftp/python/"
$GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
$GET_PIP_PATH = "C:\get-pip.py"

$OPENBLAS_BASE_URL = "https://bitbucket.org/carlkl/mingw-w64-for-python/downloads/"
$OPENBLAS_64_FILENAME = "openblas-x86-64-2014-07.7z"
$OPENBLAS_32_FILENAME = "openblas-i686-2014-07.7z"


function DownloadFile ($url, $filename, $download_folder) {
    $webclient = New-Object System.Net.WebClient
    $filepath = $download_folder + "\" + $filename
    if (Test-Path $filepath) {
        Write-Host "Reusing" $filepath
        return $filepath
    }
    # Download and retry up to 3 times in case of network transient errors.
    Write-Host "Downloading" $filename "from" $url
    $retry_attempts = 2
    for ($i=0; $i -lt $retry_attempts; $i++) {
        try {
            $webclient.DownloadFile($url, $filepath)
            break
        }
        Catch [Exception]{
            Start-Sleep 1
        }
    }
    if (Test-Path $filepath) {
       Write-Host "File saved at" $filepath
    } else {
       # Retry once to get the error message if any at the last try
       $webclient.DownloadFile($url, $filepath)
    }
    return $filepath
}


function InstallMsi ($msipath, $target_dir) {
    Write-Host "Installing" $msipath "to" $target_dir
    if (Test-Path $target_dir) {
        Write-Host $target_dir "already exists, skipping."
        return
    }
    $install_log = $msipath + ".log"
    $install_args = "/qn /log $install_log /i $msipath TARGETDIR=$target_dir INSTALLDIR=$target_dir"
    $uninstall_args = "/qn /x $msipath"
    RunCommand "msiexec.exe" $install_args
    if (-not(Test-Path $target_dir)) {
        Write-Host "$msipath seems to be installed else-where, reinstalling."
        RunCommand "msiexec.exe" $uninstall_args
        RunCommand "msiexec.exe" $install_args
    }
    if (Test-Path $target_dir) {
        Write-Host "$msipath installation complete"
    } else {
        Write-Host "Failed to install $msipath in $target_dir"
        Get-Content -Path $install_log
        Exit 1
    }
}


function DownloadPython ($python_version, $architecture, $download_folder) {
    if ($architecture -eq "32") {
        $platform_suffix = ""
    } else {
        $platform_suffix = ".amd64"
    }
    $filename = "python-" + $python_version + $platform_suffix + ".msi"
    $url = $PYTHOHN_BASE_URL + $python_version + "/" + $filename
    $msipath = DownloadFile $url $filename $download_folder
    Write-Host $msipath
    return $msipath
}


function InstallPython ($python_version, $architecture, $python_home, $download_folder) {
    Write-Host "Installing Python" $python_version "for" $architecture "bit architecture to" $python_home
    if (Test-Path $python_home) {
        Write-Host $python_home "already exists, skipping."
        return
    }
    $msipath = DownloadPython $python_version $architecture $download_folder
    InstallMsi $msipath $python_home
}

function InstallSevenZip ($target_dir, $download_folder) {
    $msipath = DownloadFile $SEVEN_ZIP_URL $SEVEN_ZIP_FILENAME $download_folder
    InstallMsi $msipath $target_dir
}


function RunCommand ($command, $command_args) {
    Write-Host $command $command_args
    Start-Process -FilePath $command -ArgumentList $command_args -Wait -Passthru
}


function InstallPip ($python_home) {
    $pip_path = $python_home + "\Scripts\pip.exe"
    $python_path = $python_home + "\python.exe"
    if (-not(Test-Path $pip_path)) {
        Write-Host "Installing pip..."
        $webclient = New-Object System.Net.WebClient
        $webclient.DownloadFile($GET_PIP_URL, $GET_PIP_PATH)
        Write-Host "Executing:" $python_path $GET_PIP_PATH
        Start-Process -FilePath "$python_path" -ArgumentList "$GET_PIP_PATH" -Wait -Passthru
    } else {
        Write-Host "pip already installed."
    }
}


function InstallOpenBLAS ($openblas_home, $architecture, $download_folder) {
    if (Test-Path $openblas_home) {
        Write-Host $openblas_home "already exists, skipping."
        return
    }

    if ( $architecture -eq "32" ) {
        $filename = $OPENBLAS_32_FILENAME
    } else {
        $filename = $OPENBLAS_64_FILENAME
    }
    $url = $OPENBLAS_BASE_URL + $filename
    $archive_path = DownloadFile $url $filename $download_folder
    Write-Host "Extracting $archive_path to $openblas_home"
    7z x -o"$openblas_home" $archive_path
}


function main () {
    # Use environment variables to pass parameters: use reasonable defaults
    # to make it easier to debug
    $download_folder = $env:DOWNLOAD_FOLDER
    if ( !$download_folder ) { $download_folder = $pwd.Path + "\Downloads"; }
    if (-not(Test-Path $download_folder)) {
        Write-Host "Creating download folder at $download_folder"
        mkdir $download_folder
    }
    $python_version = $env:PYTHON_VERSION
    if ( !$python_version ) { $python_version = "3.4.2"; }
    $python_arch = $env:PYTHON_ARCH
    if ( !$python_arch ) { $python_arch = "64"; }
    $python_home = $env:PYTHON
    if ( !$python_home ) { $python_home = "C:\Python34-x64"; }
    InstallPython $python_version $python_arch $python_home $download_folder
    InstallPip $python_home
    $sevenzip_home = $env:SEVENZIP_HOME
    if ( $sevenzip_home ) {
        InstallSevenZip $sevenzip_home $download_folder
        $env:Path = $sevenzip_home + ';' + $env:Path
    }
    $openblas_home = $env:OPENBLAS_HOME
    if ( $openblas_home ) {
        InstallOpenBLAS $openblas_home $python_arch $download_folder
    }
}

main
