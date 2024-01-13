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

# AzulEnvironment & AzulSolver

Welcome to the AzulEnvironment and AzulSolver project! This repository is the home of a unique endeavor: the first-ever (afaik) Markov Decision Process (MDP) implementation for the beloved board game, Azul. 

As a passionate Azul player and an enthusiast of artificial intelligence, I wanted to create an AI capable of surpassing my own skills in this game. This project is not just a testament to my love for Azul but also an exploration into the realms of AI and game theory.

Here, you'll find two main components:
- **AzulEnvironment**: A simulation environment that faithfully replicates the Azul game, designed for 2-4 players.
- **AzulSolver**: An AI agent, built using Proximal Policy Optimization (PPO), will hopefully master the AzulEnvironment, especially in a 2-player setting.

Whether you're an AI enthusiast, a game theory student, or just someone who loves Azul, this project has something exciting for you. Dive in and explore how AI can be trained to play and excel at one of the most popular board games of our time!
I will be updating this README as I go with more information on usage and future updates.

## Requirements

To get the most out of the AzulEnvironment and AzulSolver, you'll need to have the following libraries installed:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `pytorch`: A deep learning framework that powers the AI components of the solver.
- `pygame`: Used for creating the graphical user interface.

These libraries are essential for running the project. Make sure you have them installed before you proceed. If you're unsure how to install these, you can generally use pip, Python's package installer. For example, to install numpy, you would run:

```bash
pip install numpy
```

## Game Rules of Azul

Azul is an engaging and strategic board game where players compete to create beautiful mosaic patterns. Here's a brief overview of the game's rules:

- **Objective**: The main goal is to score the most points by creating patterns with tiles on your board.

- **Setup**: Each player has a personal board with a 5x5 grid where they place tiles. Tiles are drawn from a common pool.

- **Gameplay**:
  - **Drafting Tiles**: Players take turns drafting colored tiles from the center to their pattern lines.
  - **Tile Placement**: Once the drafting phase is over, players move the tiles from the pattern lines to their 5x5 grid.
  - **Scoring**: Points are scored based on how you place the tiles to decorate the wall. Points are awarded immediately for tiles placed, and extra points are scored for specific patterns and completed sets.
  - **Penalties**: If you collect tiles that you can't use, they count as negative points.

- **End of the Game**: The game ends after a set number of rounds, typically when at least one player has completed a horizontal line on their board.

- **Winner**: The player with the most points at the end of the game wins.

This is a simplified overview of Azul's rules. The AzulEnvironment replicates these rules in a digital format, allowing for both human and AI interaction within the game's strategic framework.
