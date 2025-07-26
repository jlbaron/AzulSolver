# AzulSolver
This project contains an Azul MDP environment for 2-4 players as well as a PPO agent to solve a 2 player game.


<h1> PROGRESS </h1>

<p> I am looking to revamp the whole environment into one compatible with industry standard reinforcement learning libraries. I want to support single agent training with sb3 against random or coded opponents and fully competitive multi agent training with ray and pettingzoo. I will start with remaking the base environment then, if needed, I will make some wrappers for the libraries. Once the sanity checks look good I will switch to agent training experiments. I will leave my initial agent here for now, it could be a useful reference for a full ppo implementation with all of the internals exposed. However, I will switch to doing agent training with the libraries once the environment is cleaned up and compliant. </p>

# AzulEnvironment & AzulSolver

Welcome to the AzulEnvironment and AzulSolver project! This repository is the home of a unique endeavor: the first-ever (afaik) Markov Decision Process (MDP) implementation for the beloved board game, Azul. 

As a passionate Azul player and an enthusiast of artificial intelligence, I wanted to create an AI capable of surpassing my own skills in this game. This project is not just a testament to my love for Azul but also an exploration into the realms of AI and game theory.

Here, you'll find two main components:
- **AzulEnvironment**: A simulation environment that faithfully replicates the Azul game, designed for 2-4 players.
- **AzulSolver**: An AI agent, built using Proximal Policy Optimization (PPO), will hopefully master the AzulEnvironment, especially in a 2-player setting.

Whether you're an AI enthusiast, a game theory student, or just someone who loves Azul, this project has something exciting for you. Dive in and explore how AI can be trained to play and excel at one of my favorite games of all time!
I will be updating this README as I go with more information on usage and future updates.

## Set up

WORK IN PROGRESS. Once I get to testing I will include the versions of libraries in the requirements file. From there set up is simple:

```
python -m venv venv

(launch environment)

pip install -r requirements.txt
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
