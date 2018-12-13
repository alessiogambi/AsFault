Param (
    [Parameter(Mandatory=$true)]$start,
    [Parameter(Mandatory=$true)]$end
)

$start..$end |
   Foreach-Object {
	$exps_dir = "./experiments/$_"
	New-Item -ItemType "directory" -Path "$exps_dir"
	Copy-Item ".asfaultenv" -Destination "$exps_dir/.asfaultenv" -Recurse
}