** General notes **:

The screen is very small, so the range of motion before the beam lands outside is quite small.
We can incorporate this physical constraints as bounds in the optimization method.

Maximum allowable movements (from screen edge-to-edge):
    - Motor translations: +/- 500 microns
    - Pitch angle: +/- 0.2 radians
    - Curvature: +/- 20 mm

---------------------------
** Optimization notes ** :

Modify the optimization "loss function" to optimize for the position of the beam
(so that it hits the center of the screen), and not the peak intensity or FWHM.
The primary objective of the focusing procedure is to ensure the beam hits the center of the screen.
We expect that the crude focusing procedure already produces a reasonably tight coherent beam,
and there is not much room for play in the beam size.

Optimizing the beam size or beam intensity can be a secondary objective.
An alternating procedure optimizing the position first, then the beam size or beam intensity would be ok.

Initial work:
    - Assume that the beam curvature and Q are ok.
    - Only adjust the horizontal and vertical motors.
    - First start with one of these parameters, then try optimizing both simultaneously.
    - Use "relative movement" for the motors.

