Param (
    [Parameter(Mandatory=$true)]$experimentsDir,
    [Parameter(Mandatory=$true)]$timeLimit
)

# This is large on purpose to let timeLimit trigger
$generations=1000

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
            $environmentDir = $_.FullName 
            "Executing experiment $experiment with environment $environmentDir"
            # Execute the experiment from the right folder
            cd C:\Users\Alessio\AsFault
            C:\Users\Alessio\AsFault\.alessio\Scripts\python.exe .\src\asfault\app.py --log $log evolve --env $environmentDir bng --generations $generations --time-limit $timeLimit --render
            cd C:\Users\Alessio\AsFault\scripts
        }
    }
}   