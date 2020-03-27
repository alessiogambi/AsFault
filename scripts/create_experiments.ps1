Param (
    [Parameter(Mandatory=$true)]$envDir,
    [Parameter(Mandatory=$true)]$experimentDir,
    [Parameter(Mandatory=$true)]$prefix,
    [Parameter(Mandatory=$true)]$start,
    [Parameter(Mandatory=$true)]$end
)

$start..$end |
   Foreach-Object {
	$exps_dir = "$experimentDir/$prefix-$_"
	New-Item -ItemType "directory" -Path "$exps_dir"
	Copy-Item $envDir -Destination "$exps_dir/$envDir" -Recurse
}