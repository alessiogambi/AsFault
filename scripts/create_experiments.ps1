Param (
    [Parameter(Mandatory=$true)]$envDir,
    [Parameter(Mandatory=$true)]$experimentDir,
    [Parameter(Mandatory=$true)]$start,
    [Parameter(Mandatory=$true)]$end
)

# $envDir=".esec-fse-20-beamng-small-env"
# $experimentDir="./FSE20-beamng-small"

# $envDir=".esec-fse-20-beamng-large-env"
# $experimentDir="./FSE20-beamng-large"

# $envDir=".esec-fse-20-beamng-small-env-with-repair"
# $experimentDir="./FSE20-beamng-small-with-repair"

#$envDir=".esec-fse-20-beamng-large-env-with-repair"
#$experimentDir="./FSE20-beamng-large-with-repair"


$start..$end |
   Foreach-Object {
	$exps_dir = "$experimentDir/$_"
	New-Item -ItemType "directory" -Path "$exps_dir"
	Copy-Item $envDir -Destination "$exps_dir/$envDir" -Recurse
}