=======
AsFault
=======

AsFault is a tool to generate test cases for self-driving cars. The tests are executed in a simulator called BeamNG_. Currently, the test cases aim at testing the "lane keeping" functionality of the self-driving cars, that is, the ability of the self-driving car of not driving off of its appropriate lane. Each test consists of one or more roads on a fixed-size map (whose size can be specified), and the goal of the self-driving AI agent is to follow a given and predetermined path from an initial position to a goal position. A good test is one that creates multiple situations where the self-driving car drives off the lane. We call such situations out-of-bounds examples (OBE).

The tests are designed and executed in different phases. The tests are first designed using poly-lines. Then AsFault creates the code needed for the simulator, BeamNG, to simulate the driving situations, execute the tests and collect, report and visualise the results (that is, the number of times the car drove off the lane).

.. _BeamNG: https://beamng.gmbh/research/