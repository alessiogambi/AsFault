# List all the experiment folders inside ./experiments
$maxSpeed=50
$generations=10
dir ./experiments -Directory |
Foreach-Object {
	$experiment = $_.FullName
    Write-Host  "$($experiment)"
	$log = "$($experiment)\experiment.log"
    $env = "$($experiment)\.asfaultenv"
    # Execute the experiment from the right folder
    cd C:\Users\Alessio\AsFault
    C:\Users\Alessio\AsFault\.alessio\Scripts\python.exe .\src\asfault\app.py --log $log evolve --env $env ext --generations $generations --render "C:\Users\Alessio\AsFault\src\deepdrive\.venv\Scripts\python.exe C:\Users\Alessio\AsFault\src\deepdrive\wrapper.py --max-speed $maxSpeed"
}
