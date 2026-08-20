import numpy as np
import torch


class PhotonBeamEncoder:
    """Builds a simplified divergent-beam / MLC-aperture mask for one VMAT control point.

    Geometry model (IEC 61217-style; the challenge JSON has no collimator or
    couch angle, so both are assumed 0):
      - Gantry rotates about the patient's superior-inferior axis (our array
        z-axis), sweeping the source through the patient's axial (x,y) plane
        at a constant z equal to the beam's iso_center z. gantry_angle=0 is
        taken as the source on the +y side of isocenter.
      - The MLC leaf-stacking direction (which of the 80 pairs) stays aligned
        with z throughout the arc; leaf open/close position ("left"/"right")
        is along the direction that rotates with the gantry.
      - mlc_left_int_mm / mlc_right_int_mm are each leaf pair's raw signed
        tip position (mm) along the leaf-motion axis, specified at the
        isocenter plane (SAD from source) and scaled linearly with depth for
        a divergent (conical) beam. The open aperture for a leaf pair is
        simply [left, right] -- NOT [-left, right]. An earlier version of
        this code incorrectly negated `left` (treating it as a
        distance-from-center magnitude); real data shows both fields can be
        negative or positive independently (e.g. left=-51, right=-33 is a
        real, open, off-center aperture), and negating `left` produced an
        invalid (inverted) range for ~9% of real leaf/control-point entries
        vs ~0.02% with the direct-coordinate interpretation used now. See
        PROGRESS_LOG.md for the full investigation.

    Calibrated against real data: predicted mask vs. actual dose overlap was
    checked across all 6 beams of both downloaded patients (1ABB006,
    1ABB011; 600 wide-aperture control points total, see PROGRESS_LOG.md)
    to pick the rotation sign convention (`rotation_sign`) -- a naive
    dose-centroid check was tried first but was too noisy (body-wide
    scatter dominates a simple centroid), so overlap with the real per-CP
    dose grid was used instead as ground truth. rotation_sign=-1.0 won
    553/600 control points outright (11 for +1.0, 36 ties) and is the
    default. Narrow/near-closed apertures (<15mm average leaf opening)
    give near-zero overlap under either sign -- expected, since body-wide
    scatter swamps a sliver-thin direct beam in a total-sum overlap
    metric, not a sign-convention problem. **This calibration was originally
    run against the buggy [-left, right] aperture formula above.** Re-checked
    after the fix on 204 wide-aperture control points (both patients, all 6
    beams): rotation_sign=-1.0 still wins, even more decisively (195/204 vs
    the original 553/600 rate). The formula fix itself was validated the
    same way: mean dose-overlap went from 3.6% (buggy [-left,right]) to
    77.3% (fixed [left,right]), and the fixed formula won outright on all
    204/204 control points checked -- see PROGRESS_LOG.md. Treat this mask
    as a coarse geometric prior (no penumbra/scatter modeling), not
    dose-accurate ground truth.
    """

    LEAF_WIDTH_MM = 5.0  # Elekta Agility 160-leaf MLC, standard leaf width at isocenter
    NUM_LEAVES = 80

    def __init__(self, volume_shape, voxel_spacing, origin, rotation_sign=-1.0):
        """
        Args:
            volume_shape: (D, H, W) i.e. (z, y, x) voxel counts
            voxel_spacing: (dz, dy, dx) in mm
            origin: (x, y, z) physical mm of voxel (0,0,0) -- sitk convention
            rotation_sign: +1.0 or -1.0, calibrated gantry rotation handedness
        """
        self.volume_shape = volume_shape
        self.voxel_spacing = voxel_spacing
        self.rotation_sign = rotation_sign
        # _coords_xyz is built origin-free (as if origin=(0,0,0)) so it only
        # depends on (volume_shape, voxel_spacing) -- identical for every
        # patient once CenterCropOrPad normalizes everyone to the same
        # target_shape. This lets Trainer cache ONE encoder for the whole
        # run instead of one per patient. `origin` becomes a cheap (3,)
        # offset applied to the beam source instead (see _beam_geometry) --
        # mathematically identical, since rel = (coords+origin) - source is
        # the same as coords - (source-origin). Set via `set_origin()` per
        # sample rather than baked into a full-volume tensor: caching one
        # coordinate grid per patient blew a 40GB GPU on Task 3 (proton)'s
        # much larger 512x512x176 grid (~553MB/patient x 67 train patients
        # -> ~37GB) partway through epoch 0 -- see PROGRESS_LOG.md.
        self._coords_xyz = self._build_physical_coords()
        self.set_origin(origin)

    def set_origin(self, origin):
        """Updates the per-patient origin offset in place -- cheap (3,)
        tensor, not a full-volume rebuild. Call before encode()/beam_depth()
        for each new patient."""
        self.origin = torch.as_tensor(origin, dtype=torch.float32, device=self._coords_xyz.device)

    def to(self, device):
        """Moves both GPU-resident tensors together. Callers used to move
        only `_coords_xyz` directly (`encoder._coords_xyz = encoder.
        _coords_xyz.to(device)`) -- that missed `origin` once it became a
        separate tensor (the origin-factoring refactor above), causing a
        cuda:0/cpu device-mismatch crash the first time `origin` wasn't
        (0,0,0). Use this instead of touching the tensors directly."""
        self._coords_xyz = self._coords_xyz.to(device)
        self.origin = self.origin.to(device)
        return self

    def _build_physical_coords(self):
        d, h, w = self.volume_shape
        dz, dy, dx = self.voxel_spacing
        z = torch.arange(d, dtype=torch.float32) * dz
        y = torch.arange(h, dtype=torch.float32) * dy
        x = torch.arange(w, dtype=torch.float32) * dx
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1)  # (D, H, W, 3) in x,y,z order, origin-free

    def _beam_geometry(self, beam_params):
        """Shared setup for encode() and beam_depth(): source position and
        the three axes (beam/leaf-motion/leaf-stack) for this control point.
        """
        device = self._coords_xyz.device
        iso = torch.as_tensor(beam_params["iso_center"], dtype=torch.float32, device=device) - self.origin
        sad = float(beam_params["SAD"])
        theta = self.rotation_sign * np.deg2rad(float(beam_params["gantry_angle"]))

        beam_axis = torch.tensor([-np.sin(theta), -np.cos(theta), 0.0], device=device)  # source -> iso
        leaf_motion_axis = torch.tensor([np.cos(theta), -np.sin(theta), 0.0], device=device)  # rotates with gantry
        leaf_stack_axis = torch.tensor([0.0, 0.0, 1.0], device=device)  # fixed, S-I

        source = iso - sad * beam_axis
        rel = self._coords_xyz - source  # (D, H, W, 3)
        return rel, beam_axis, leaf_motion_axis, leaf_stack_axis, sad

    def beam_depth(self, beam_params):
        """Per-voxel depth (mm) along the central beam axis from the source.

        Exposed as a public method (rather than inlined only in `encode()`)
        because `src/evaluation/metrics.py`'s IDD Curve Distance metric needs
        the same depth values to bin dose by depth along the beam.
        """
        rel, beam_axis, _, _, _ = self._beam_geometry(beam_params)
        return (rel * beam_axis).sum(-1)

    def encode(self, beam_params, body_mask=None):
        """
        Args:
            beam_params: dict with 'gantry_angle', 'iso_center' (3,), 'SAD',
                'mlc_left_int_mm' (80,), 'mlc_right_int_mm' (80,) -- the flat
                per-control-point dict produced by DoseRAD2026Dataset.
            body_mask: unused -- accepted only so callers can invoke
                PhotonBeamEncoder and ProtonBeamEncoder through the same
                signature (ProtonBeamEncoder needs it, this doesn't).

        Returns:
            torch.Tensor: (1, D, H, W) binary aperture mask.
        """
        rel, beam_axis, leaf_motion_axis, leaf_stack_axis, sad = self._beam_geometry(beam_params)

        t = (rel * beam_axis).sum(-1)  # depth along central axis from source
        lateral = (rel * leaf_motion_axis).sum(-1)  # in-plane, rotates with gantry
        cc = (rel * leaf_stack_axis).sum(-1)  # cranio-caudal (z)

        scale = torch.clamp(t / sad, min=1e-6)
        lateral_iso = lateral / scale
        cc_iso = cc / scale

        leaf_idx = torch.round(cc_iso / self.LEAF_WIDTH_MM + (self.NUM_LEAVES / 2 - 0.5)).long()
        valid_leaf = (leaf_idx >= 0) & (leaf_idx < self.NUM_LEAVES) & (t > 0)
        leaf_idx_clamped = leaf_idx.clamp(0, self.NUM_LEAVES - 1)

        left = torch.as_tensor(beam_params["mlc_left_int_mm"], dtype=torch.float32)
        right = torch.as_tensor(beam_params["mlc_right_int_mm"], dtype=torch.float32)
        left_at_voxel = left[leaf_idx_clamped]
        right_at_voxel = right[leaf_idx_clamped]

        # mlc_left_int_mm / mlc_right_int_mm are each leaf pair's raw signed
        # tip position along the leaf-motion axis (not a magnitude from
        # center) -- confirmed empirically: negating left produced an
        # inverted (invalid) [right, left] range for ~9% of real
        # leaf/control-point entries, vs ~0.02% using the raw values
        # directly. See PROGRESS_LOG.md for the numbers.
        in_aperture = (lateral_iso >= left_at_voxel) & (lateral_iso <= right_at_voxel)
        mask = (valid_leaf & in_aperture).float().unsqueeze(0)
        return mask


class ProtonBeamEncoder:
    """Builds a single-Gaussian pencil-beam-scanning dose proxy for one proton
    beamlet (one energy layer of one scanned spot/ray).

    Geometry model: unlike PhotonBeamEncoder (which must reconstruct the
    source position from gantry_angle + iso_center + SAD, since the photon
    JSON never gives it directly), each proton *ray* already carries its own
    explicit `ray_source` and `ray_target` (mm, same coordinate frame as
    ct_origin/iso_center) -- confirmed against real data (1ABB006.json): the
    two points are ~1000mm apart along the beam axis, consistent with a
    clinical source-to-axis distance. The beam axis for a ray is simply
    normalize(ray_target - ray_source); every beamlet on that ray shares the
    axis and differs only in `energy` (MeV), which sets how far into the
    patient the Bragg peak lands. Each (beam, ray, beamlet) triple is one
    independent training sample -- see DoseRAD2026Dataset(task="proton").

    Dose-proxy shape: the challenge reference doc describes the proton beam
    model itself as an "energy-dependent single-Gaussian approximation", so
    this encoder mirrors that structure directly rather than approximating
    a different (e.g. double-Gaussian/Bortfeld) model:
      - Longitudinal (along-axis) profile: a Gaussian centered on the proton
        range R(energy), from the Bragg-Kleeman range-energy relation for
        water (R[mm] = 10 * 0.0022 * E[MeV]^1.77, accurate to ~1% over
        ~10-250 MeV -- covers the reference doc's stated 31.7-200.8 MeV
        range) and Bortfeld's (1997, Med Phys) mono-energetic range
        straggling fit (sigma_R[mm] = 10 * 0.012 * R[cm]^0.935).
      - Lateral (off-axis) profile: a Gaussian whose sigma grows with depth
        via the Highland multiple-Coulomb-scattering formula, using water's
        radiation length (X0=360.8mm) and relativistic proton kinematics
        (rest mass 938.272 MeV) to get beta*p*c at this beamlet's energy.
      - Zero dose beyond ~3 longitudinal sigma past the range -- protons
        deposit negligible dose past the Bragg peak; that sharp distal
        falloff is the defining feature of proton therapy, unlike photon's
        gradual exponential attenuation.

    Like PhotonBeamEncoder, this is a coarse physical prior fed to the model
    as an input channel, not a claim of dose-accurate ground truth -- the
    network learns the real dose shape from the Monte Carlo targets during
    training.

    Range is measured from where the ray enters the body, not from
    `ray_source` itself: `ray_source` is a virtual point ~1000mm away in air
    (same SAD convention as photon), and the Bragg-Kleeman range only
    applies to the water-equivalent path *inside* the patient. Using raw
    source distance instead produced an all-zero mask for every real
    beamlet checked (range ~19mm for a 45.6MeV beamlet vs. ~1000mm from
    source to the target/iso point -- five sigma outside the longitudinal
    Gaussian, underflowing to exactly 0). `encode()` therefore requires a
    `body_mask` and finds each ray's entry depth by taking the shallowest
    body-mask voxel within `NEAR_AXIS_MM` of the ray line -- this is a
    per-ray geometric property independent of energy, so it's valid to
    reuse across every beamlet on the same ray.

    Sanity-checked against real data (1ABB006, 5 beamlets spanning beams
    0/1/5, energies 45.6-140.4 MeV): predicted peak voxel lands within
    3-17mm of the actual MC dose grid's peak voxel in every case -- a
    reasonable result for a coarse physical prior with no scatter/CT
    density modeling, not a claim of dose accuracy. See PROGRESS_LOG.md.
    """

    WATER_X0_MM = 360.8        # radiation length of water
    PROTON_MASS_MEV = 938.272  # proton rest mass energy
    NEAR_AXIS_MM = 2.0         # lateral tolerance for "on the ray" when ray-marching for body entry

    def __init__(self, volume_shape, voxel_spacing, origin):
        """
        Args:
            volume_shape: (D, H, W) i.e. (z, y, x) voxel counts
            voxel_spacing: (dz, dy, dx) in mm
            origin: (x, y, z) physical mm of voxel (0,0,0) -- sitk convention
        """
        self.volume_shape = volume_shape
        self.voxel_spacing = voxel_spacing
        # _coords_xyz is built origin-free -- see PhotonBeamEncoder's
        # __init__ docstring for why (this is what made a per-patient
        # cache of it OOM a 40GB GPU on this task's much larger grid).
        # `origin` is a cheap (3,) offset applied to the ray source instead.
        self._coords_xyz = self._build_physical_coords()
        self.set_origin(origin)

    def set_origin(self, origin):
        """Updates the per-patient origin offset in place -- cheap (3,)
        tensor, not a full-volume rebuild. Call before encode() for each
        new patient."""
        self.origin = torch.as_tensor(origin, dtype=torch.float32, device=self._coords_xyz.device)

    def to(self, device):
        """Moves both GPU-resident tensors together -- see
        PhotonBeamEncoder.to()'s docstring for why this exists instead of
        moving `_coords_xyz` directly."""
        self._coords_xyz = self._coords_xyz.to(device)
        self.origin = self.origin.to(device)
        return self

    def _build_physical_coords(self):
        d, h, w = self.volume_shape
        dz, dy, dx = self.voxel_spacing
        z = torch.arange(d, dtype=torch.float32) * dz
        y = torch.arange(h, dtype=torch.float32) * dy
        x = torch.arange(w, dtype=torch.float32) * dx
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack([xx, yy, zz], dim=-1)  # (D, H, W, 3) in x,y,z order, origin-free

    @staticmethod
    def _range_mm(energy_mev):
        return 10.0 * 0.0022 * energy_mev ** 1.77

    @staticmethod
    def _range_straggling_mm(range_mm):
        range_cm = range_mm / 10.0
        return 10.0 * 0.012 * range_cm ** 0.935

    @classmethod
    def _highland_sigma_mm(cls, depth_mm, energy_mev):
        """Highland-approximation lateral sigma (mm) at along-axis depth `depth_mm`."""
        e_tot = energy_mev + cls.PROTON_MASS_MEV
        pc = (e_tot ** 2 - cls.PROTON_MASS_MEV ** 2) ** 0.5
        beta = pc / e_tot

        depth_clamped = torch.clamp(depth_mm, min=1.0)  # avoid log(0)/div-by-0 at the entrance
        frac = depth_clamped / cls.WATER_X0_MM
        theta0 = (14.1 / (beta * pc)) * torch.sqrt(frac) * (1.0 + torch.log10(frac) / 9.0)
        theta0 = torch.clamp(theta0, min=0.0)  # Highland's log term can dip slightly negative at very shallow depth
        sigma = depth_clamped * theta0
        return torch.clamp(sigma, min=1.0)  # 1mm floor -- avoids a zero-width singularity as depth->0

    def _entry_depth(self, depth, lateral, body_mask):
        """Depth (mm from source) where this ray first crosses into the body.

        A per-ray geometric property, computed once per ray (not per
        beamlet/energy) and reused for every beamlet on that ray -- see
        class docstring. Falls back to depth=0 (i.e. no offset applied) if
        no body-mask voxel is found near the ray, which degrades gracefully
        to the old (wrong) behavior rather than crashing.
        """
        near_axis = lateral < self.NEAR_AXIS_MM
        body = body_mask.to(torch.bool)
        if body.dim() == 4:
            body = body.squeeze(0)
        valid = near_axis & body & (depth > 0)
        if valid.any():
            return depth[valid].min()
        return torch.zeros((), device=depth.device)

    def encode(self, beam_params, body_mask=None):
        """
        Args:
            beam_params: dict with 'ray_source' (3,), 'ray_target' (3,),
                'energy' (scalar) -- the flat per-beamlet dict produced by
                DoseRAD2026Dataset for task="proton".
            body_mask: (1, D, H, W) or (D, H, W) tensor from BodyMask/
                BodyMaskMR, same volume as this encoder's coordinate grid.
                Required to place the Bragg peak correctly -- see class
                docstring for why raw source distance doesn't work.

        Returns:
            torch.Tensor: (1, D, H, W) single-Gaussian pencil-beam dose proxy.
        """
        device = self._coords_xyz.device
        source = torch.as_tensor(beam_params["ray_source"], dtype=torch.float32, device=device)
        target = torch.as_tensor(beam_params["ray_target"], dtype=torch.float32, device=device)
        energy = float(beam_params["energy"])

        axis = target - source
        axis = axis / axis.norm()

        rel = self._coords_xyz - (source - self.origin)  # (D, H, W, 3), both sides in the origin-free frame
        depth = (rel * axis).sum(-1)  # along-axis distance from source
        perp = rel - depth.unsqueeze(-1) * axis
        lateral = perp.norm(dim=-1)

        entry_depth = self._entry_depth(depth, lateral, body_mask) if body_mask is not None else torch.zeros((), device=device)
        in_patient_depth = torch.clamp(depth - entry_depth, min=0.0)

        r = self._range_mm(energy)
        sigma_r = self._range_straggling_mm(r)
        sigma_lat = self._highland_sigma_mm(in_patient_depth, energy)
        peak_depth = entry_depth + r

        longitudinal = torch.exp(-0.5 * ((depth - peak_depth) / sigma_r) ** 2)
        lateral_falloff = torch.exp(-0.5 * (lateral / sigma_lat) ** 2)

        mask = longitudinal * lateral_falloff
        beyond_range_or_behind_entry = (depth > (peak_depth + 3.0 * sigma_r)) | (depth < entry_depth)
        mask = mask.masked_fill(beyond_range_or_behind_entry, 0.0)
        return mask.unsqueeze(0)
