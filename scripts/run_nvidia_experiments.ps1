Param (
    [Parameter(Mandatory=$true)]$experimentsDir,
    [Parameter(Mandatory=$true)]$timeLimit
)

# This is large on purpose to let timeLimit trigger
$generations=1000

$wrapperScript="C:\LuganoPassau\.venv\Scripts\python.exe C:\LuganoPassau\asfault\nvidia_ai.py"

dir $experimentsDir -Directory |
Foreach-Object {
	$experiment = $_.FullName

    # Check if this experiments already ran, hence produced a log file
    $log = "$($experiment)\experiment.log"
    
    if (Test-Path -Path $log -PathType Leaf){
        "Experiment $experiment already run. Skip it"
    } else {
        # Look for env folders and execute them
        ls $experiment .*-env -Directory |
        Foreach-Object {
            # This should be exctracted from the execution.json file !s
            $environmentDir = $_.FullName
            $json=(Get-Content $environmentDir\cfg\execution.json | ConvertFrom-Json)
            $maxSpeed=$json.speed_limit
            $wrapperScript="$wrapperScript --max-speed $maxSpeed"

            "Executing experiment $experiment with environment $environmentDir"
            # Execute the experiment from the right folder
            cd C:\Users\Alessio\AsFault\
            C:\Users\Alessio\AsFault\.alessio\Scripts\python.exe .\src\asfault\app.py --log $log evolve --env $environmentDir ext --generations $generations --time-limit $timeLimit --render $wrapperScript
            cd C:\Users\Alessio\AsFault\scripts
        }
    }
}   










