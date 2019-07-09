=======
AsFault
=======

AsFault is a tool to generate test cases for self-driving cars. The tests are executed in a simulator called BeamNG_. Currently, the test cases aim at testing the "lane keeping" functionality of the self-driving cars, that is, the ability of the self-driving car of driving on its own lane. Each test consists of one or more roads on a fixed-size map (whose size can be specified), and the goal of the self-driving AI agent is to follow a given and predetermined path from an initial position to a goal position. A good test is one that creates multiple situations where the self-driving car drives off the lane. We call such situations out-of-bounds examples (OBE).

The tests are designed and executed in different phases. The tests are first designed using poly-lines. Then AsFault creates the code needed for the simulator, BeamNG, to simulate the driving situations, execute the tests and collect, report and visualise the results (that is, the number of times the car drove off the lane).

There are several details that need to be explained. In particular, we need to specify how the poly-lines are exactly generated, which path is chosen for the AI to follow and how to create effective test cases (that is, test cases that cause the self-driving car to drive off its lane).

-----------------
Example Execution
-----------------

You can run AsFault using e.g. :code:`asfault evolve bng --seed 1234 --budget 2 --show --render`. You should run the mentioned command from inside the `AsFault` folder (otherwise you will get errors). See the file `src/asfault/app.py`_ for more details regarding these options. The options are specified using the Python module click_.


.. _BeamNG: https://beamng.gmbh/research/
.. _click: https://click.palletsprojects.com/en/7.x/
.. _src/asfault/app.py: src/asfault/app.py