Param(
	[string]$experiments,
	[string]$fitness,
	[string]$bounds,
	[string]$risk,
	[string]$join,
	[string]$userpath,
	[string]$port,
	[string]$population,
	[string]$budget
)

$now = Get-Date -format s
$now = $now -replace ":","-"

$experiments = "$($experiments)\$($now)"

$exp_cmd = "C:\Users\Alessio\AsFault\.venv\Scripts\python.exe .\src\asfault\app.py write_experiment_envs"

$fitness.Split(';') |
Foreach-Object {
	$exp_cmd = "$($exp_cmd) --fitness $($_)"
}

$bounds.Split(';') |
Foreach-Object {
	$exp_cmd = "$($exp_cmd) --bounds $($_)"
}

$risk.Split(';') |
Foreach-Object {
	$exp_cmd = "$($exp_cmd) --risk $($_)"
}

$join.Split(';') |
Foreach-Object {
	$exp_cmd = "$($exp_cmd) --join $($_)"
}

$userpath.Split(';') |
Foreach-Object {
	$exp_cmd = "$($exp_cmd) --userpath $($_)"
}

$port.Split(';') |
Foreach-Object {
	$exp_cmd = "$($exp_cmd) --port $($_)"
}

$exp_cmd = "$($exp_cmd) --population $($population) --budget $($budget) $($experiments)"
Invoke-Expression $exp_cmd

dir $experiments -Directory |
Foreach-Object {
	$exps_dir = $_.FullName
	$script = "$($exps_dir)\run_experiments.ps1"
	start powershell $script
}