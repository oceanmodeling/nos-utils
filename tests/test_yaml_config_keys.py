"""System-specific values live in yaml/presets, not in system-name tests. MJ (11/18/26)"""

import numpy as np
import pytest

from nos_utils.config import ForcingConfig

yaml = pytest.importorskip("yaml")

PDY = "20261001"

# The block the nos-workflow ATL yaml carries (keys already present there are repeated here). MJ (11/18/26)
ATL_YAML = """
system: {name: stofs_3d_atl_ufs}
grid: {domain: {lon_min: -98.5035, lon_max: -52.4867, lat_min: 7.347, lat_max: 52.5904}}
model: {physics: {nws: 4}}
prep:
  ops_bad_day_checks: true
forcing:
  atmospheric:
    gfs: {ops_timeline: true}
    hrrr: {rotate_winds: false, blend_rotate_winds: false, blend_weight: 0.99, small_grib: true}
  river: {ops_static_files: true}
  ocean:
    obc: {ops_timeline: true, ssh_hold_first_record: true}
    nudging: {enabled: true, dt: 21600}
    adt:
      enabled: true
      weight_file: stofs_3d_atl_adt_weight.nc
      datum_offset: -0.45
      roi: [-62.5, -51.5, 7.0, 54.0]
"""

KEYS = ("adt_roi", "adt_datum_offset", "nudging_dt", "obc_ssh_hold_first_record", "obc_ops_timeline",
        "gfs_ops_timeline", "hrrr_rotate_winds", "datm_rotate_hrrr_winds", "datm_hrrr_weight",
        "hrrr_small_grib", "river_ops_static", "ops_bad_day_checks", "adt_enabled")


def _load(tmp_path, text, name="x"):
    p = tmp_path / (name + ".yaml")
    p.write_text(text)
    return ForcingConfig.from_yaml(p, pdy=PDY, cyc=12)


@pytest.mark.parametrize("factory", ["for_stofs_3d_atl", "for_stofs_3d_atl_ufs"])
def test_atl_presets_keep_their_values(factory):
    c = getattr(ForcingConfig, factory)(PDY, 12)
    assert c.adt_roi == (-62.5, -51.5, 7.0, 54.0)
    assert c.adt_datum_offset == -0.45 and c.nudging_dt == 21600.0
    assert c.obc_ssh_hold_first_record is True and c.obc_ops_timeline is True
    assert c.gfs_ops_timeline is True and c.hrrr_rotate_winds is False
    assert c.datm_rotate_hrrr_winds is False and c.datm_hrrr_weight == 0.99
    assert c.hrrr_small_grib is True and c.river_ops_static is True
    assert c.ops_bad_day_checks is True
    assert str(c.adt_weight_file) == "stofs_3d_atl_ufs.adt_weight.nc"


@pytest.mark.parametrize("factory", ["for_stofs_3d_pac", "for_stofs_3d_pac_ufs"])
def test_pacific_presets_keep_their_values(factory):
    c = getattr(ForcingConfig, factory)(PDY, 12)
    assert c.adt_datum_offset == -0.4715517 and c.adt_roi is None and c.nudging_dt == 86400.0
    assert c.obc_ops_timeline is False and c.ops_bad_day_checks is False


def test_atl_yaml_block_matches_the_atl_presets(tmp_path):
    y, p = _load(tmp_path, ATL_YAML), ForcingConfig.for_stofs_3d_atl_ufs(PDY, 12)
    for k in KEYS:
        assert getattr(y, k) == getattr(p, k), k


def test_atl_name_alone_turns_nothing_on(tmp_path):
    c = _load(tmp_path, "system: {name: stofs_3d_atl_ufs}\n")
    assert c.obc_ssh_hold_first_record is False and c.obc_ops_timeline is False
    assert c.gfs_ops_timeline is False and c.hrrr_rotate_winds is True and c.hrrr_small_grib is False
    assert c.datm_hrrr_weight == 1.0 and c.river_ops_static is False and c.ops_bad_day_checks is False
    assert c.adt_weight_file is None and c.adt_roi is None and c.adt_enabled is False


def test_adt_enabled_requires_datum_offset(tmp_path):
    body = "system: {name: anything}\nforcing:\n  ocean:\n    adt: {enabled: true}\n"
    with pytest.raises(ValueError, match="datum_offset"):
        _load(tmp_path, body)
    ok = _load(tmp_path, body.replace("true}", "true, datum_offset: -0.3}"))
    assert ok.adt_datum_offset == -0.3 and ok.adt_roi is None
    off = _load(tmp_path, "system: {name: anything}\nforcing:\n  ocean:\n    adt: {enabled: false}\n")
    assert off.adt_datum_offset == 0.0


@pytest.mark.parametrize("roi", ["[1, 2, 3]", "[5, 1, 0, 1]", "[0, 1, 4, 3]", "abc"])
def test_adt_roi_is_validated(tmp_path, roi):
    body = ("system: {name: anything}\nforcing:\n  ocean:\n    adt: {enabled: true, datum_offset: 0.0, roi: %s}\n" % roi)
    with pytest.raises(ValueError, match="adt.roi"):
        _load(tmp_path, body)


def test_adt_roi_selection_is_config_driven(tmp_path):
    from nos_utils.forcing.adt import ADTBlender
    base = dict(lon_min=-98.0, lon_max=-52.0, lat_min=7.0, lat_max=52.0, pdy=PDY, cyc=12)
    missing = tmp_path / "none.nc"
    unset = ADTBlender(ForcingConfig(**base), tmp_path)
    assert unset._roi(missing) == (-99.0, -51.0, 6.0, 53.0)
    boxed = ADTBlender(ForcingConfig(adt_roi=(-62.5, -51.5, 7.0, 54.0), **base), tmp_path)
    assert boxed._roi(missing) == (-62.5, -51.5, 7.0, 54.0)
    assert boxed._dest_bounds(missing) == (-62.5, -51.5, 7.0, 54.0)


def _nudge_len(tmp_path, dt, ops_tl):
    from unittest.mock import patch
    from netCDF4 import Dataset
    from tests.test_nudging import TestNudgingWithSyntheticRTOFS as T
    from nos_utils.forcing.nudging import NudgingProcessor
    from nos_utils.io.schism_vgrid import SchismVgrid
    t = T()
    n = 5
    lons = np.linspace(-77.0, -73.0, n); lats = np.linspace(30.0, 34.0, n)
    t._make_gr3(tmp_path / "n.gr3", n, lons, lats, np.ones(n))
    t._make_grid(tmp_path / "hgrid.ll", n, lons, lats, np.full(n, 100.0))
    t._make_vgrid_simple(tmp_path / "vgrid.in", nvrt=10)
    d = tmp_path / "rtofs.20260331"
    d.mkdir()
    for fhr in range(12, 133, 6):
        t._make_rtofs_3d(d / f"rtofs_glo_3dz_f{fhr:03d}_6hrly_hvr_US_east.nc", 1, 5, (-80, -70), (28, 36), 20)
    cfg = ForcingConfig(lon_min=-80.0, lon_max=-70.0, lat_min=28.0, lat_max=36.0, pdy="20260401", cyc=12,
                        nowcast_hours=24, forecast_hours=48, nudging_enabled=True, n_levels=10,
                        grid_file=tmp_path / "hgrid.ll", nudging_dt=dt, obc_ops_timeline=ops_tl)
    proc = NudgingProcessor(cfg, tmp_path, tmp_path / "o", nudge_weight_file=tmp_path / "n.gr3",
                            rtofs_input_path=tmp_path, phase="nowcast")
    with patch.object(proc, "_load_vgrid", return_value=SchismVgrid.read(tmp_path / "vgrid.in")), \
            patch("nos_utils.forcing.rtofs.RTOFSProcessor.validate_file_size", return_value=True):
        res = proc.process()
    assert res.success, res.errors
    with Dataset(tmp_path / "o" / "TEM_nu.nc") as ds:
        return ds.dimensions["time"].size


@pytest.mark.parametrize("ops_tl", [True, False])
def test_nudging_step_comes_from_config_not_the_ops_flag(tmp_path, ops_tl):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    six = _nudge_len(tmp_path / "a", 21600.0, ops_tl)
    three = _nudge_len(tmp_path / "b", 10800.0, ops_tl)
    assert three > six


def test_ops_timeline_requires_nudging_dt(tmp_path):
    body = "system: {name: x}\nforcing:\n  ocean:\n    obc: {ops_timeline: true}\n"
    with pytest.raises(ValueError, match="nudging.dt"):
        _load(tmp_path, body)
    ok = _load(tmp_path, body + "    nudging: {dt: 21600}\n")
    assert ok.obc_ops_timeline is True and ok.nudging_dt == 21600.0


def test_adt_enabled_string_false_is_false(tmp_path):
    body = "system: {name: x}\nforcing:\n  ocean:\n    adt: {enabled: 'false'}\n"
    assert _load(tmp_path, body).adt_enabled is False


def _scheduled(tmp_path, monkeypatch, cfg, run_name):
    from nos_utils.forcing.base import ForcingResult
    from nos_utils.orchestrator import PrepOrchestrator
    calls = []

    def stub(name):
        def f(self, *a, **k):
            calls.append(name)
            return ForcingResult(success=True, source=name.upper())
        return f

    for n in ("hotstart", "gfs", "hrrr", "nwm", "tidal", "rtofs", "nudging", "river_ob_clim",
              "st_lawrence", "dynamic_adjust", "surface_restore", "param_nml", "datm",
              "ufs_config", "wave_boundary"):
        monkeypatch.setattr(PrepOrchestrator, "_run_" + n, stub(n), raising=False)
    monkeypatch.setattr(PrepOrchestrator, "_write_time_markers", lambda *a, **k: None, raising=False)
    paths = {"output": tmp_path / "out", "nwm": tmp_path / "nwm", "gfs": tmp_path / "gfs",
             "hrrr": tmp_path / "hrrr"}
    res = PrepOrchestrator(cfg, paths, run_name=run_name, skip_legacy=False).run("nowcast")
    return calls, res


@pytest.mark.parametrize("enabled,run_name", [
    (True, "stofs_3d_atl_ufs"), (True, "secofs_ufs"), (False, "stofs_3d_pac_ufs")])
def test_nwm_scheduled_only_when_enabled(tmp_path, monkeypatch, enabled, run_name):
    cfg = ForcingConfig.for_stofs_3d_atl_ufs(PDY, 12, nwm_enabled=enabled)
    assert ("nwm" in _scheduled(tmp_path, monkeypatch, cfg, run_name)[0]) is enabled


def test_disabled_nwm_is_not_a_critical_failure(tmp_path, monkeypatch):
    cfg = ForcingConfig.for_stofs_3d_atl_ufs(PDY, 12, nwm_enabled=False, ops_bad_day_checks=False,
                                             critical_sources=["GFS", "NWM", "TIDAL"])
    calls, res = _scheduled(tmp_path, monkeypatch, cfg, "stofs_3d_pac_ufs")
    assert "nwm" not in calls and res.success


def test_secofs_preset_schedules_nwm(tmp_path, monkeypatch):
    assert "nwm" in _scheduled(tmp_path, monkeypatch, ForcingConfig.for_secofs(PDY, 12), "secofs")[0]


@pytest.mark.parametrize("val,expect", [("true", True), ("false", False), (None, True)])
def test_river_enabled_parses(tmp_path, val, expect):
    river = "" if val is None else "  river: {enabled: %s}\n" % val
    assert _load(tmp_path, "system: {name: x}\nforcing:\n" + river + "  tidal: {}\n").nwm_enabled is expect


def test_river_disabled_with_ob_clim_enabled_is_valid(tmp_path):
    c = _load(tmp_path, "system: {name: x}\nforcing:\n  river:\n    enabled: false\n    ob_clim: {enabled: true}\n")
    assert c.nwm_enabled is False and c.river_ob_clim_enabled is True


@pytest.mark.parametrize("factory", ["for_stofs_3d_pac", "for_stofs_3d_pac_ufs"])
def test_pacific_presets_disable_nwm(factory):
    assert getattr(ForcingConfig, factory)(PDY, 12).nwm_enabled is False
    assert ForcingConfig.for_stofs_3d_atl(PDY, 12).nwm_enabled is True
    assert ForcingConfig.for_secofs(PDY, 12).nwm_enabled is True
