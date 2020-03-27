Param (
    [Parameter(Mandatory=$true)]$envDir,
    [Parameter(Mandatory=$true)]$experimentsDir,
    [Parameter(Mandatory=$true)]$timeLimit
)

# $envDir=".esec-fse-20-beamng-small-env"
# $experimentsDir="./FSE20-beamng-small"

# $envDir=".esec-fse-20-beamng-small-env-with-repair"
# $experimentsDir="./FSE20-beamng-small-with-repair"

# $envDir=".esec-fse-20-beamng-large-env"
# $experimentsDir="./FSE20-beamng-large"

# $envDir=".esec-fse-20-beamng-large-env-with-repair"
# $experimentsDir="./FSE20-beamng-large-with-repair"

$generations=100

dir $experimentsDir -Directory |
Foreach-Object {
	$experiment = $_.FullName

    # Check if this experiments already ran, hence produced a log file
    $log = "$($experiment)\experiment.log"
    
    if (Test-Path -Path $log -PathType Leaf){
        "Experiment $experiment already run. Skip it"
    } else {
        "Executing experiment $experiment"
        $env = "$($experiment)\$envDir"

        # Execute the experiment from the right folder
        cd C:\Users\Alessio\AsFault
        C:\Users\Alessio\AsFault\.alessio\Scripts\python.exe .\src\asfault\app.py --log $log evolve --env $env bng --generations $generations --time-limit $timeLimit --render
        cd C:\Users\Alessio\AsFault\scripts
    }
}   