Param (
    [Parameter(Mandatory=$true)]$experimentsDir,
    [Parameter(Mandatory=$true)]$timeLimit
)

# This is large on purpose to let timeLimit trigger
$generations=3000

# $wrapperScript="C:\LuganoPassau\.venv\Scripts\python.exe C:\LuganoPassau\asfault\nvidia_ai.py"
$wrapperScript="C:\illumination\.venv\Scripts\python.exe C:\illumination\ai_drive\dave2.py " #--debug"

dir $experimentsDir -Directory |
Foreach-Object {
	$experiment = $_.FullName
    $wrapperScript="C:\illumination\.venv\Scripts\python.exe C:\illumination\ai_drive\dave2.py "
    # Use the folder name to configure nvidia output-to 
    $wrapperScript="$wrapperScript --output-to $_"

    # Check if this experiments already ran, hence produced a log file
    $log = "$($experiment)\experiment.log"
    
    if (Test-Path -Path $log -PathType Leaf){
        "Experiment $experiment already run. Skip it"
    } else {
        "Running Experiment $experiment"

        # Look for env folders and execute them
        ls $experiment .*env -Directory |
        Foreach-Object {
            # This should be exctracted from the execution.json file !
            $environmentDir = $_.FullName
            # TODO Use default configuration for the moment, then possibly update the model
            #$wrapperScript="$wrapperScript --max-speed $maxSpeed"

            "Executing experiment $experiment with environment $environmentDir"
            
            # Execute the experiment from the right folder
            #cd C:\Users\Alessio\AsFault\
            python.exe C:\modified-asfault\AsFault\src\asfault\app.py --log $log evolve --env $environmentDir ext --generations $generations --time-limit $timeLimit --use-simulation-time --render BeamNG
            #$wrapperScript
            #cd C:\Users\Alessio\AsFault\scripts
        }
    }
}   










