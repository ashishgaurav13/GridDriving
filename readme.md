## CarGridDriving

<img src="images/circuit.png" width=300/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="images/closeup.png" width=300/>

### Graph Construction

This simulator is an extension of the OpenAI Gym [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) environment. The original environment constructs a heavily morphed circlular track and the goal is to loop around on this track. The simulator used in this project extends this environment by constructing a (h, w) grid, where each edge appears with probability p.

Then d edges are deleted from this graph and the road edges are constructed if the graph is connected. Connectedness ensures that if there are multiple vehicles, any vehicle can reach any place and hence the vehicles can collide. This process is repeated for as long as it takes to meet the graph constraints on the number of 1-edge, 2-edge, 3-edge and 4-edge junctions, and connectedness.

### Rendering

Finally, once a connected constraint satisfying graph is created, the road polygons are drawn on a square playfield of size P. For every 3-edge and 4-edge intersections, traffic lights are placed in coordination. The traffic lights follow schedule to cycle between the available signs for every edge and only allow for consistent and safe intersection crossing.

The simulator can be started with n vehicles, and for every step of this environment, each of the vehicles get a viewport capture from their perspective, with only the traffic lights relevant to them showing up in the capture.

### Features
- Traffic schedule and rendering support for 3-edge and 4-edge intersections
- Constraint support for 1-edge, 2-edge, 3-edge, 4-edge intersections
- Finer polygons can be drawn by changing parameters
- Generalized to n vehicles, each vehicle viewer only sees relevant information
- Localization information

### Things to do

- [ ] Update for sparse rewards
- [ ] Train on options graph
- [ ] Update documentation in code

### License

AGPLv3, check out licenses folder.
