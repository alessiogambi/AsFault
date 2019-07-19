=======
AsFault
=======

AsFault is a tool to generate test cases for self-driving cars. The tests are executed in a simulator called BeamNG_. Currently, the test cases aim at testing the "lane keeping" functionality of the self-driving cars, that is, the ability of the self-driving car of driving on its own lane. Each test consists of one or more roads on a fixed-size map (whose size can be specified), and the goal of the self-driving AI agent is to follow a given and predetermined path from an initial position to a goal position. A good test is one that creates multiple situations where the self-driving car drives off the lane. We call such situations out-of-bounds examples (OBE).

The tests are designed and executed in different phases. The tests are first designed using poly-lines. Then AsFault creates the code needed for the simulator, BeamNG, to simulate the driving situations, execute the tests and collect, report and visualise the results (that is, the number of times the car drove off the lane).

There are several details that need to be explained. In particular, we need to specify how the poly-lines are exactly generated, which path is chosen for the AI to follow and how to create effective test cases (that is, test cases that cause the self-driving car to drive off its lane).

-------------
Prerequisites
-------------

- Windows
- SVN
- Python 3
- pip
- virtualenv (or Anaconda)

----------------------------
BeamNG.research Installation
----------------------------

Given that AsFault uses the simulator BeamNG.research, we first need to download it. To do it, we can use the command: :code:`svn checkout https://projects.beamng.com/svn/research` (which thus requires SVN to be installed on your system). I recommend that you checkout this repository inside :code:`C:\Users\<You>\Documents`. Then you need to set the environment variable :code:`BNG_HOME` to point to the folder containing the executable of BeamNG.research. To do that, you can follow the instructions in one of the answers to this question https://superuser.com/q/949560. For example, if you downloaded BeamNG.research to :code:`C:\Users\<You>Documents\beamng\research`, where :code:`research` is the name of the downloaded repository, then you should have the environment variable :code:`BNG_HOME` that points to this path.

--------------------
AsFault Installation
--------------------

You first need to clone AsFault using e.g. :code:`https://github.com/Signaltonsalat/AsFault.git`. Then I recommend that you install AsFault in a virtual environment so that you do not pollute your global Python environment. In the following instructions, I assume that you're inside a virtual environment.

If you're using virtualenv (rather than Anaconda), you will likely encounter a problem while installing the package AsFault in your virtual environment. Specifically, it is possible you will encounter a problem related to the installation of the required (by AsFault) package "shapely", something like "OSError: [WinError 126] module could not be found" or "Command "python setup.py egg_info" failed with error code 1 in ...shapely". In that case, I followed the solution provided in this answer https://gis.stackexchange.com/a/62931. Essentially, you should download the wheel file of shapely that corresponds to your installed version of Python from https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely, then you should install shapely using e.g. :code:`pip install Shapely‑1.6.4.post2‑cp36‑cp36m‑win32.whl` (where the file :code:`Shapely‑1.6.4.post2‑cp36‑cp36m‑win32.whl` should be under the current directory), where, in my case, I am using Python 3.6 (a 32-bit version), but, of course, you will download and install the version of shapely that corresponds to your Python version. If you don't install the version of shapely that matches your Python distribution, you will likely encounter an incompatibility error.

If you're using an Anaconda environment, you should be able to install shapely using :code:`conda install shapely`.

You can now install the AsFault package. Go inside the cloned repo, and then type :code:`pip install -e .`, which essentially installs AsFault in your current environment in an editable mode.

-----------------
Example Execution
-----------------

You can run AsFault using e.g. :code:`asfault evolve bng --seed 1234 --budget 2 --show --render`. You should run the mentioned command from inside the :code:`AsFault` folder (otherwise you will get errors). See the file `src/asfault/app.py`_ for more details regarding these options. The options are specified using the Python module click_.


.. _BeamNG: https://beamng.gmbh/research/
.. _click: https://click.palletsprojects.com/en/7.x/
.. _src/asfault/app.py: src/asfault/app.py