# List all the experiment folders inside ./experiments
$generations=100

# Stop the simulation after 1h
$time_limit=3600

dir ./FSE20 -Directory |
Foreach-Object {
	$experiment = $_.FullName
    Write-Host  "$($experiment)"
	$log = "$($experiment)\experiment.log"
    $env = "$($experiment)\.esec-fse-20-beamng-small-env"
    # Execute the experiment from the right folder
    cd C:\Users\Alessio\AsFault
    C:\Users\Alessio\AsFault\.alessio\Scripts\python.exe .\src\asfault\app.py --log $log evolve --env $env bng --generations $generations --time-limit $time_limit --render
    cd C:\Users\Alessio\AsFault\scripts
}   