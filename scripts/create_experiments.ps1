Param (
    [Parameter(Mandatory=$true)]$start,
    [Parameter(Mandatory=$true)]$end
)

$start..$end |
   Foreach-Object {
	$exps_dir = "./FSE20/$_"
	New-Item -ItemType "directory" -Path "$exps_dir"
	Copy-Item ".esec-fse-20-beamng-small-env" -Destination "$exps_dir/.esec-fse-20-beamng-small-env" -Recurse
}