%% AAA-LS Laplace Solver Benchmark
% Implements the AAA-LS algorithm from Costa & Trefethen (2023) for
% solving Laplace equation on polygonal domains, including multiply-connected.
% Sweeps AAA tolerance and records timing + solution on delta=0.05 grid.
%
% Requires: aaa.m (Chebfun standalone), VAorthog.m, VAeval.m

addpath(fileparts(mfilename('fullpath')));

delta = 0.05;
n_repeats = 5;
n_poly = 20;        % polynomial degree for LS fit
n_pts_per_side = 300; % sample points per boundary side

results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10];

%% ========== Unit Square ==========
fprintf('\n=== AAA-LS: Unit Square ===\n');
w_sq = [0, 1, 1+1i, 1i].';

Z_sq = [];
for k = 1:4
    Z_sq = [Z_sq; cluster_points(w_sq(k), w_sq(mod(k,4)+1), n_pts_per_side)];
end

H_sq = zeros(size(Z_sq));
H_sq(1:n_pts_per_side) = 1.0;  % hot_bottom

xmin=0; xmax=1; ymin=0; ymax=1;
nx = ceil((xmax-xmin)/delta);
ny = ceil((ymax-ymin)/delta);
[Xg, Yg] = meshgrid(linspace(xmin+0.5*delta, xmax+0.5*delta, nx), ...
                     linspace(ymin+0.5*delta, ymax+0.5*delta, ny));
Zg_sq = Xg + 1i*Yg;

fid_csv = fopen(fullfile(results_dir, 'aaals_unit_square.csv'), 'w');
time_hdr = '';
for r = 1:n_repeats
    time_hdr = [time_hdr sprintf(',time%d', r)];
end
fprintf(fid_csv, 'tol%s,maxerr\n', time_hdr);

for ti = 1:length(tols)
    tol = tols(ti);
    fprintf('  tol = %.0e: ', tol);
    try
        times = zeros(1, n_repeats);
        n_poles = 0;
        for r = 1:n_repeats
            tic;
            [u_func, ~, pol_out] = solve_aaals_simple(w_sq, Z_sq, H_sq, tol, n_poly);
            times(r) = toc;
            n_poles = length(pol_out);
        end
        mean_time = mean(times);
        sol = u_func(Zg_sq);
        maxerr = max(abs(sol(:)));  % rough
        fprintf('mean=%.4f s, std=%.4f s, poles=%d\n', mean_time, std(times), n_poles);

        fname = fullfile(results_dir, sprintf('aaals_unit_square_tol%.0e.dat', tol));
        fid = fopen(fname, 'w');
        for j = 1:ny
            for i = 1:nx
                fprintf(fid, '%.15e %.15e %.15e\n', Xg(j,i), Yg(j,i), sol(j,i));
            end
        end
        fclose(fid);
        fprintf(fid_csv, '%.0e', tol);
        for r = 1:n_repeats
            fprintf(fid_csv, ',%.6e', times(r));
        end
        fprintf(fid_csv, ',%.6e\n', maxerr);
    catch ME
        fprintf('FAILED: %s\n', ME.message);
    end
end
fclose(fid_csv);

%% ========== L-shape ==========
fprintf('\n=== AAA-LS: L-shape ===\n');
w_ls = [0, 2, 2+1i, 1+1i, 1+2i, 2i].';

Z_ls = [];
nw_ls = length(w_ls);
for k = 1:nw_ls
    Z_ls = [Z_ls; cluster_points(w_ls(k), w_ls(mod(k,nw_ls)+1), n_pts_per_side)];
end

H_ls = zeros(size(Z_ls));
H_ls(1:n_pts_per_side) = 1.0;  % hot_bottom

xmin=0; xmax=2; ymin=0; ymax=2;
nx = ceil((xmax-xmin)/delta);
ny = ceil((ymax-ymin)/delta);
[Xg, Yg] = meshgrid(linspace(xmin+0.5*delta, xmax+0.5*delta, nx), ...
                     linspace(ymin+0.5*delta, ymax+0.5*delta, ny));
Zg_ls = Xg + 1i*Yg;

fid_csv = fopen(fullfile(results_dir, 'aaals_l_shape.csv'), 'w');
time_hdr = '';
for r = 1:n_repeats
    time_hdr = [time_hdr sprintf(',time%d', r)];
end
fprintf(fid_csv, 'tol%s,maxerr\n', time_hdr);

for ti = 1:length(tols)
    tol = tols(ti);
    fprintf('  tol = %.0e: ', tol);
    try
        times = zeros(1, n_repeats);
        n_poles = 0;
        for r = 1:n_repeats
            tic;
            [u_func, ~, pol_out] = solve_aaals_simple(w_ls, Z_ls, H_ls, tol, n_poly);
            times(r) = toc;
            n_poles = length(pol_out);
        end
        mean_time = mean(times);
        sol = u_func(Zg_ls);
        fprintf('mean=%.4f s, std=%.4f s, poles=%d\n', mean_time, std(times), n_poles);

        fname = fullfile(results_dir, sprintf('aaals_l_shape_tol%.0e.dat', tol));
        fid = fopen(fname, 'w');
        for j = 1:ny
            for i = 1:nx
                fprintf(fid, '%.15e %.15e %.15e\n', Xg(j,i), Yg(j,i), sol(j,i));
            end
        end
        fclose(fid);
        fprintf(fid_csv, '%.0e', tol);
        for r = 1:n_repeats
            fprintf(fid_csv, ',%.6e', times(r));
        end
        fprintf(fid_csv, ',%.6e\n', maxerr);
    catch ME
        fprintf('FAILED: %s\n', ME.message);
    end
end
fclose(fid_csv);

%% ========== Square with Hole (doubly-connected) ==========
fprintf('\n=== AAA-LS: Square with Hole ===\n');
w_outer = [0, 1, 1+1i, 1i].';
w_hole_v = [0.3+0.3i, 0.7+0.3i, 0.7+0.7i, 0.3+0.7i].';

% Outer boundary
Z_swh = [];
for k = 1:4
    Z_swh = [Z_swh; cluster_points(w_outer(k), w_outer(mod(k,4)+1), n_pts_per_side)];
end
% Hole boundary (reversed orientation)
for k = 1:4
    Z_swh = [Z_swh; cluster_points(w_hole_v(mod(k,4)+1), w_hole_v(k), n_pts_per_side)];
end

H_swh = zeros(size(Z_swh));
H_swh(1:n_pts_per_side) = 1.0;  % hot_bottom on outer

xmin=0; xmax=1; ymin=0; ymax=1;
nx = ceil((xmax-xmin)/delta);
ny = ceil((ymax-ymin)/delta);
[Xg, Yg] = meshgrid(linspace(xmin+0.5*delta, xmax+0.5*delta, nx), ...
                     linspace(ymin+0.5*delta, ymax+0.5*delta, ny));
Zg_swh = Xg + 1i*Yg;

fid_csv = fopen(fullfile(results_dir, 'aaals_square_with_hole.csv'), 'w');
time_hdr = '';
for r = 1:n_repeats
    time_hdr = [time_hdr sprintf(',time%d', r)];
end
fprintf(fid_csv, 'tol%s,maxerr\n', time_hdr);

for ti = 1:length(tols)
    tol = tols(ti);
    fprintf('  tol = %.0e: ', tol);
    try
        times = zeros(1, n_repeats);
        n_poles = 0;
        for r = 1:n_repeats
            tic;
            [u_func, ~, pol_out] = solve_aaals_doubly(w_outer, w_hole_v, Z_swh, H_swh, tol, n_poly);
            times(r) = toc;
            n_poles = length(pol_out);
        end
        mean_time = mean(times);
        sol = u_func(Zg_swh);
        fprintf('mean=%.4f s, std=%.4f s, poles=%d\n', mean_time, std(times), n_poles);

        fname = fullfile(results_dir, sprintf('aaals_square_with_hole_tol%.0e.dat', tol));
        fid = fopen(fname, 'w');
        for j = 1:ny
            for i = 1:nx
                fprintf(fid, '%.15e %.15e %.15e\n', Xg(j,i), Yg(j,i), sol(j,i));
            end
        end
        fclose(fid);
        fprintf(fid_csv, '%.0e', tol);
        for r = 1:n_repeats
            fprintf(fid_csv, ',%.6e', times(r));
        end
        fprintf(fid_csv, ',%.6e\n', maxerr);
    catch ME
        fprintf('FAILED: %s\n', ME.message);
    end
end
fclose(fid_csv);

fprintf('\nAAA-LS benchmark done. Results saved to %s\n', results_dir);
exit;

%% ===================== LOCAL FUNCTIONS =====================

function Z = cluster_points(z1, z2, np)
% Clustered points on line segment z1->z2 with tanh clustering near ends
    s = tanh(linspace(-14,14,np)');
    s = (s - s(1))/(s(end) - s(1));
    Z = z1 + s*(z2 - z1);
end

function [u, elapsed, pol_out] = solve_aaals_simple(w, Z, H, tol, n_poly)
% AAA-LS for simply-connected polygon
    inpoly = @(z,w) inpolygon(real(z),imag(z),real(w),imag(w));
    nw = length(w);

    % Local AAA fits near each corner
    pol_out = [];
    for k = 1:nw
        dists = abs(Z - w(k));
        min_dists = min(abs(Z - w.'), [], 2);
        ii = find(dists == min_dists);
        if length(ii) < 3, continue; end
        try
            [~, polk] = aaa(H(ii), Z(ii), 'tol', tol, 'cleanup', 0);
            polk_out = polk(~inpoly(polk, w));
            pol_out = [pol_out; polk_out(:)];
        catch
        end
    end

    if isempty(pol_out)
        % Fallback: exponentially clustered poles near corners
        for k = 1:nw
            kn = mod(k, nw) + 1;
            dir_out = exp(1i*(angle(w(kn) - w(k)) + pi));
            for j = 1:20
                pol_out = [pol_out; w(k) + 0.5*exp(-j) * dir_out];
            end
        end
        pol_out = pol_out(~inpoly(pol_out, w));
    end

    pol = pol_out(:).';

    % Time the LS solve
    tic;
    d = min(abs(Z - pol), [], 1);
    [Hes, P] = VAorthog(Z, n_poly);
    Q = d ./ (Z - pol);
    A = [real(P) real(Q) -imag(P) -imag(Q)];
    c = reshape(A \ H, [], 2) * [1; 1i];
    elapsed = toc;

    u = @(z) real(reshape([VAeval(z(:), Hes) d./(z(:)-pol)] * c, size(z)));
end

function [u, elapsed, pol_out] = solve_aaals_doubly(w_out, w_hole, Z, H, tol, n_poly)
% AAA-LS for doubly-connected polygon (outer + one hole)
    inpoly = @(z,w) inpolygon(real(z),imag(z),real(w),imag(w));
    zc = mean(w_hole);
    all_corners = [w_out(:); w_hole(:)];

    % Artificial BC for pole generation (sqrt of product of distances to corners)
    H_art = ones(size(Z));
    for k = 1:length(all_corners)
        H_art = H_art .* sqrt(abs(Z - all_corners(k)));
    end

    pol_out = [];
    for k = 1:length(all_corners)
        dists = abs(Z - all_corners(k));
        min_dists = min(abs(Z - all_corners.'), [], 2);
        ii = find(dists == min_dists);
        if length(ii) < 3, continue; end
        try
            [~, polk] = aaa(H_art(ii), Z(ii), 'tol', tol, 'cleanup', 0);
            in_outer = inpoly(polk, w_out);
            in_hole = inpoly(polk, w_hole);
            valid = ~in_outer | in_hole;
            pol_out = [pol_out; polk(valid)];
        catch
        end
    end

    if isempty(pol_out)
        nw_out = length(w_out);
        nw_hole = length(w_hole);
        for k = 1:nw_out
            dir_out = exp(1i*(angle(mean(w_out) - w_out(k)) + pi));
            for j = 1:20
                pol_out = [pol_out; w_out(k) + 0.5*exp(-j) * dir_out];
            end
        end
        for k = 1:nw_hole
            dir_in = exp(1i*angle(zc - w_hole(k)));
            for j = 1:15
                pol_out = [pol_out; w_hole(k) + 0.2*exp(-j) * dir_in];
            end
        end
        in_outer = inpoly(pol_out, w_out);
        in_hole = inpoly(pol_out, w_hole);
        pol_out = pol_out(~in_outer | in_hole);
    end

    pol = pol_out(:).';
    n_hole_poly = 10;

    tic;
    d = min(abs(Z - pol), [], 1);
    [Hes, P] = VAorthog(Z, n_poly);

    Qh = zeros(length(Z), n_hole_poly);
    for j = 1:n_hole_poly
        Qh(:, j) = 1 ./ (Z - zc).^j;
    end
    Q = d ./ (Z - pol);
    log_col = log(abs(Z - zc));

    A = [real(P) real(Q) real(Qh) log_col ...
         -imag(P) -imag(Q) -imag(Qh) zeros(size(log_col))];
    c_full = A \ H;
    elapsed = toc;

    n_real = size(P,2) + size(Q,2) + n_hole_poly + 1;
    cr = c_full(1:n_real);
    ci = c_full(n_real+1:end);

    u = @(z) eval_doubly(z, Hes, pol, d, zc, n_hole_poly, cr, ci);
end

function val = eval_doubly(z, Hes, pol, d, zc, n_hole_poly, cr, ci)
    z_col = z(:);
    P = VAeval(z_col, Hes);
    Q = d ./ (z_col - pol);
    Qh = zeros(length(z_col), n_hole_poly);
    for j = 1:n_hole_poly
        Qh(:, j) = 1 ./ (z_col - zc).^j;
    end
    log_col = log(abs(z_col - zc));

    real_part = [real(P) real(Q) real(Qh) log_col] * cr;
    imag_part = [-imag(P) -imag(Q) -imag(Qh) zeros(size(log_col))] * ci;
    val = reshape(real_part + imag_part, size(z));
end
