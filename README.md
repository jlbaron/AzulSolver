# AzulSolver
This project contains an Azul MDP environment for 2-4 players as well as a PPO agent to solve a 2 player game.


<h1> PROGRESS </h1>

<p> Environment: complete and mostly optimized </p>
<p> Solver: starting first training runs </p>
<p> Visualization: functional but ugly </p>
<p> Extras: different observation option in environment, testing for 3-4 players, more training options, testing </p>
<p> I will also need to optimize this whole thing with more vectorization and possible gpu support </p>

<p> 1/2/2023: began testing environment, more work towards agent train loop </p>
<p> 1/3/2023: progress testing environment, train loop nearly complete, planning visualizations </p>
<p> 1/5/2023: took a major detour into pygame board representations. Nearly useable and will be helpful in finishing up the environment.</p>
<p> 1/7/2023: first successful play through with random moves, can't guarantee there are no bugs and it is unoptimized but it works! </p>
<p> 1/8/2023: cleaning up everything. optimized azul environment with some more list comprehension </p>
<p> 1/10/2023: can now visualize 2-4 players but found a bug in environment with more than 2 players</p>
<p> 1/11/2023: improved visualization, cleaned some code </p>