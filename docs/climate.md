# Climate and volumetric rendering

The climate is a deterministic procedural model shared by the ecosystem and the renderer. It is not a fluid simulation. Regional fields describe moisture, convection, upper-air cloud cover and geothermal water supply. The CPU evaluates these fields directly; the GPU reads a quantised, filtered texture generated from the same function at startup. The weather field is stable across visual noise seeds so ecological conditions and rendered weather agree.

Aur drives a 40-minute activity cycle and widespread low-altitude lift. Equatorial ascent strengthens convection. Above the gardens, flow turns toward the nearest pole and accelerates; polar subsidence dries the lower atmosphere. Altitude reduces pressure and temperature, while approaching the polar emitters increases temperature and light. Moisture determines condensation height. Instability and moisture together produce deep convection, precipitation potential and an ice fraction. Rainfall values currently inform the climate model; falling precipitation particles are not rendered.

Clouds occupy a continuous three-dimensional density field with four overlapping supports:

| Formation | Altitude | Conditions |
| --- | --- | --- |
| Garden cumulus | Condensation base to roughly 2–8 km | Moist rising air |
| Middle layers | 5.5–8.7 km | Widespread moisture |
| High ice clouds | 11–20.5 km | Upper-air moisture, including otherwise dry regions |
| Convective towers | Condensation base to roughly 22–26 km | Wet unstable regions, isolated tower footprints and spreading upper anvils |

Each altitude advects with its local circulation, producing wind shear. Cloud coverage, condensation height and density vary continuously between regions, including clear, high-only and deeply overcast conditions. Fine erosion remains inside the coarse cloud hull. Two shadow probes estimate optical depth; three scattering octaves reuse that estimate without additional shadow rays.

Aur uses broad warped density folds, finer filaments and emission from depth. Eruptions exist in a world-space cell field, independent of camera position. Each is a continuous circular curve: its analytic distance supplies the volumetric cross-section, and its angular coordinate drives travelling energy. There are no curve segments or uploaded eruption point lists. Four neighbouring cells are evaluated to preserve curves across cell boundaries; sparse activation and jittered centres avoid a regular field of identical arches. Pole beams similarly use continuous distance from the planet's rotation axis, evaluated using the original planet-relative ray point.

The marcher intersects volume bounds, skips gaps, probes coarse cloud hulls and switches to fine integration before entering occupied intervals. It increases sample spacing with distance, integrates extinction exponentially and stops once transmission is negligible. Cloud shadow work is restricted to samples containing clouds. A geometry-aware spatial filter and temporal accumulation reduce stochastic volume noise while retaining opaque silhouettes. Gaussian eruption and beam fields are density functions, not conservative signed distance fields: their distance functions can guide sampling, but their nonzero halos must still be integrated. Procedural fields have effectively unlimited placement; floating-point precision, curvature, atmospheric extinction and the finite sample budget limit practical visibility.

## References

- [Skybolt: Rendering Planetwide Volumetric Clouds](https://prograda.com/2021/07/28/rendering-planetwide-volumetric-clouds-in-skybolt/): hull/fine marching, erosion and multiple-scattering approximations.
- [Guerrilla: Nubis³, SIGGRAPH 2023](https://www.advances.realtimerendering.com/s2023/index.html): modern three-dimensional cloud modelling and distance-field acceleration. A voxel distance hierarchy remains a possible future improvement; this implementation does not build one.
- [EA: Filter-adapted spatiotemporal sampling](https://www.ea.com/news/spatio-temporal-sampling): research into sampling tailored to temporal reconstruction. The renderer currently retains its existing temporal sampling sequence.

Use `--benchmark --scene cloud-high`, `cloud-tower`, `cloud-interior`, `clear`, `equator`, `sunrise`, `polar` or `aur-ejection` for repeatable views. `--animate --time 20 --size 2048x1080` captures three frames while the fields evolve. GPU performance comparisons should run one benchmark at a time.
