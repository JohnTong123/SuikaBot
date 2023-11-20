# SuikaBot
AI Algorithm and Simulation for playing Suika Game aka watermelon game 

The first steps to take, was identifying how to simulate the game itself.

Decided on tick based simulation, with impulses.

After observations, we determined the SuikaGame was based around fruits with circular hitboxes, which, once they interact with fruits of the same type, they generate a new fruit of a greater type at the point of intersection (which is also the average of the center position of the fruits.)