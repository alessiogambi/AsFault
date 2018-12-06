.\.roadgenv\Scripts\activate

$experiments = "{{experiments}}"
$budget = "{{budget}}"

dir $experiments -Directory |
Foreach-Object {
	$experiment = $_.FullName
	$log = "$($experiment)\experiment.log"
	$out = "$($experiment)\result.json"
	python .\roadgen\app.py --log $log evo --env $experiment --flush-output experiment --budget $budget
}
