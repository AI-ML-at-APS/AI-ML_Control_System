## TODO

Here are the next things to be tried:
 
1) Separate the optimization in two steps:
    + Coarse alignement $\rightarrow$ H and V directions can be treated separately, motors can use coarse steps, optimzation stops when 
      - $centroid$ is in $\pm n$ microns and 
      - $\sigma$ $\pm m$ microns 
    ($n,m$ to be determined, let’s start with $\pm5$ and $\pm2.5$, respectively)
   + Fine alignement $\rightarrow$ all motors together, steps use motor resolutions, the optimization stops when centroid and sigma are inside a much stringent acceptance range.

2) Add a simpler bender simulation, without writing files: `bender=3`: 
    less realistic but accurate enough and faster    
 
I spoke with Xianbo, and we are actually capable of reproducing the 34-ID KB at 28-ID. Pitch and bender are already available for both the mirrors, and we can work on adding the translation. We have long beamtimes in both November and December (a total of 5 weeks) to play as much as we want. It would be amazing to collect data of both the optimizer and the ML-based controller running together. It is worth noting that both the benders (H and V) are a shared motor between optimizer and ML. It would be interesting to see how to make this interaction work.


Saugat can keep working on 34-ID, but I will start creating the libraries for 28-ID asap. The beam is not very coherent and we don’t have coherence slits, so the ray-tracing will be faster.