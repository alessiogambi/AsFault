import json
import os
import shutil
import time


class BeamNGMainFolder:
    def __init__(self, path):
        self.path = os.path.realpath(path)

    def write_items(self, content):
        with open(os.path.join(self.path, 'items.level.json'), 'w') as f:
            f.write(content)


class MapFolder:
    def __init__(self, path):
        self.path = path
        self.tig_version_json_path = os.path.join(path, 'tig-version.json')

    def exists(self):
        return os.path.exists(self.path)

    def same_version(self, other_map: 'MapFolder'):
        self_ver = self.version_info()['version']
        other_ver = other_map.version_info()['version']
        return self_ver == other_ver

    def version_info_exists(self):
        return os.path.exists(self.tig_version_json_path)

    def version_info(self):
        with open(self.tig_version_json_path, 'r') as f:
            return json.load(f)

    def tig_version_json_path(self):
        return self.path + '/'

    def delete_all_map(self):
        print(f'Removing [{self.path}]')
        shutil.rmtree(self.path, ignore_errors=True)

        # sometimes rmtree fails to remove files
        for tries in range(20):
            if os.path.exists(self.path):
                time.sleep(0.1)
                shutil.rmtree(self.path, ignore_errors=True)

        if os.path.exists(self.path):
            shutil.rmtree(self.path)

    def generated(self):
        return BeamNGMainFolder(os.path.join(self.path, r'main/MissionGroup/generated'))


class LevelsFolder:
    def __init__(self, path):
        self.path = os.path.realpath(path)

    def ensure_folder_exists(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_map(self, map_name: str):
        return MapFolder(os.path.join(self.path, map_name))


class Maps:
    beamng_map: MapFolder
    source_map: MapFolder

    def __init__(self):
        self.beamng_levels = LevelsFolder(os.path.join(os.environ['USERPROFILE'], r'Documents/BeamNG.research/levels'))
        self.source_levels = LevelsFolder(os.getcwd()+'/levels_template')
        self.source_map = self.source_levels.get_map('tig')
        self.beamng_map = self.beamng_levels.get_map('tig')
        self.never_logged_path = True

    def print_paths(self):
        print('beamng_levels', self.beamng_levels.path)
        print('source_levels', self.source_levels.path)

    def install_map_if_needed(self):
        if self.never_logged_path:
            self.never_logged_path = False
            # print(f'BeamNG userpath levels is [{self.beamng_levels.path}]')

        self.beamng_levels.ensure_folder_exists()

        if self.beamng_map.exists():
            if not self.beamng_map.version_info_exists():
                print(f'Warning! The folder [{self.beamng_map.path}] does not look like a map of tig project.\n'
                      f'It does not contains the distinctive file [{self.beamng_map.tig_version_json_path}]')
                print('Stopping execution')
                exit(1)
            else:
                if not self.beamng_map.same_version(self.source_map):
                    print(f'Maps have different version information. '
                          f'Do you want to remove all {self.beamng_map.path} folder and copy it anew?'
                          f'.\nType yes to accept, no to keep it as it is')
                    while True:
                        resp = input('>')
                        if resp in ['yes', 'no']:
                            break
                        print('Type yes or no')
                    if resp == 'yes':
                        self.beamng_map.delete_all_map()

        if not self.beamng_map.exists():
            print(f'Copying from [{self.source_map.path}] to [{self.beamng_map.path}]')
            shutil.copytree(src=self.source_map.path, dst=self.beamng_map.path)


global maps
maps = Maps()

if __name__ == '__main__':
    maps.install_map_if_needed()
