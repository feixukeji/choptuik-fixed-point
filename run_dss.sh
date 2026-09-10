#!/bin/bash
# Ran as: sbatch -p day -c 16 --mem=64G -t 16:00:00
cd "$(dirname "$(readlink -f "$0")")"
export PYTHONPATH=src
V=${PYTHON:-python}
case "$1" in
  seeds) for f in G1 T1 O1 G3; do $V src/dss_seedgen.py $f & done; wait ;;
  spec)  $V src/dss_sweep.py spec ;;
  robust) $V src/dss_sweep.py robust ;;
  seedsfix) for f in G3 O1; do $V src/dss_seedgen.py $f & done; wait ;;
  depth) for f in G1 T1 O1 G3; do $V src/dss_seedtest.py $f & done; wait ;;
  cheb) for n in 96 112 128 144 160; do $V src/dss_spec_cheb.py $n & done; wait ;;
  chebhi) for n in 176 192 208 224; do $V src/dss_spec_cheb2.py one $n & done; wait ;;
  chebeps) for n in 128 160; do $V src/dss_spec_cheb2.py eps $n & done; wait ;;
  # eps control at resolutions that actually resolve lam1 (one job per
  # (N,eps): dn.jacobian is a serial loop over columns).
  chebeps2) for n in 176 224; do for e in 1e-5 1e-4; do
      $V -c "import dss_spec_cheb2 as m; m.respec($n, $e)" & done; done; wait ;;
  chebxmax) for x in 2.5 3.0 5.0 6.0; do $V src/dss_spec_cheb2.py xmax 176 $x & done; wait ;;
  # domain scan at MATCHED near-horizon node spacing: h(X_h) ~ 0.0267 for
  # every Xmax, so that Xmax moves and the resolution at the horizon does
  # not.  Reference is (N=224, Xmax=4); (N=176, Xmax=3) already matches.
  chebmatch) $V -c "import dss_hires as hr; hr.run(262, Xmax=5.0, src='out/dss/cheb_N224_X5.npz')" ;;
  chebmatch6) $V -c "import dss_hires as hr; hr.run(298, Xmax=6.0, src='out/dss/cheb_N256_X6.npz')" ;;
  hiN)   OMP_NUM_THREADS=1 $V src/dss_hires.py N ;;
  hiX)   OMP_NUM_THREADS=1 $V src/dss_hires.py X ;;

  lam1)  OMP_NUM_THREADS=1 $V src/dss_lam1_direct.py N=176 n=2400
         OMP_NUM_THREADS=1 $V src/dss_lam1_direct.py N=176 n=1200
         OMP_NUM_THREADS=1 $V src/dss_lam1_direct.py N=224 n=2400 ;;
  *) echo "usage: $0 {seeds|spec|robust}"; exit 1 ;;
esac
