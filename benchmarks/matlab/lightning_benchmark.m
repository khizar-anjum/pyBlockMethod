%% Lightning Laplace Solver Benchmark
% Solves Laplace equation on unit_square and l_shape with hot_bottom BC.
% Sweeps tolerance parameter and records timing + solution on delta=0.05 grid.
% Outputs CSV files for each geometry with columns: tol, time_seconds, maxerr
% and .dat files with solution evaluated on the matching grid.

% Add current directory to path (for laplace.m)
addpath(fileparts(mfilename('fullpath')));

delta = 0.05;
n_repeats = 5; % median of 5 runs for timing

% Tolerance sweep
tols = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10];

results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

%% ========== Unit Square ==========
fprintf('\n=== Unit Square ===\n');
P_square = [0, 1, 1+1i, 1i]; % vertices as complex numbers (CCW)
g_square = [1, 0, 0, 0];     % hot_bottom: u=1 on edge 0->1, u=0 elsewhere

% Grid matching Volkov (delta=0.05)
xmin=0; xmax=1; ymin=0; ymax=1;
nx = ceil((xmax-xmin)/delta);
ny = ceil((ymax-ymin)/delta);
[Xg, Yg] = meshgrid(linspace(xmin+0.5*delta, xmax+0.5*delta, nx), ...
                     linspace(ymin+0.5*delta, ymax+0.5*delta, ny));
Zg_square = Xg + 1i*Yg;

fid_csv = fopen(fullfile(results_dir, 'lightning_unit_square.csv'), 'w');
time_hdr = '';
for r = 1:n_repeats
    time_hdr = [time_hdr sprintf(',time%d', r)];
end
fprintf(fid_csv, 'tol%s,maxerr\n', time_hdr);

for ti = 1:length(tols)
    tol = tols(ti);
    fprintf('  tol = %.0e: ', tol);

    % Time the solve (n_repeats runs)
    times = zeros(1, n_repeats);
    for r = 1:n_repeats
        tic;
        [u, maxerr] = laplace(P_square, g_square, 'tol', tol, 'noplots');
        times(r) = toc;
    end
    fprintf('mean=%.4f s, std=%.4f s, maxerr=%.2e\n', mean(times), std(times), maxerr);

    % Evaluate on grid
    sol = real(u(Zg_square(:)));
    sol = reshape(sol, ny, nx);

    % Save solution
    fname = fullfile(results_dir, sprintf('lightning_unit_square_tol%.0e.dat', tol));
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
end
fclose(fid_csv);

%% ========== L-shape ==========
fprintf('\n=== L-shape ===\n');
P_lshape = [0, 2, 2+1i, 1+1i, 1+2i, 2i]; % vertices as complex numbers (CCW)
g_lshape = [1, 0, 0, 0, 0, 0];            % hot_bottom: u=1 on edge 0->2

% Grid
xmin=0; xmax=2; ymin=0; ymax=2;
nx = ceil((xmax-xmin)/delta);
ny = ceil((ymax-ymin)/delta);
[Xg, Yg] = meshgrid(linspace(xmin+0.5*delta, xmax+0.5*delta, nx), ...
                     linspace(ymin+0.5*delta, ymax+0.5*delta, ny));
Zg_lshape = Xg + 1i*Yg;

fid_csv = fopen(fullfile(results_dir, 'lightning_l_shape.csv'), 'w');
time_hdr = '';
for r = 1:n_repeats
    time_hdr = [time_hdr sprintf(',time%d', r)];
end
fprintf(fid_csv, 'tol%s,maxerr\n', time_hdr);

for ti = 1:length(tols)
    tol = tols(ti);
    fprintf('  tol = %.0e: ', tol);

    times = zeros(1, n_repeats);
    for r = 1:n_repeats
        tic;
        [u, maxerr] = laplace(P_lshape, g_lshape, 'tol', tol, 'noplots');
        times(r) = toc;
    end
    fprintf('mean=%.4f s, std=%.4f s, maxerr=%.2e\n', mean(times), std(times), maxerr);

    sol = real(u(Zg_lshape(:)));
    sol = reshape(sol, ny, nx);

    fname = fullfile(results_dir, sprintf('lightning_l_shape_tol%.0e.dat', tol));
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
end
fclose(fid_csv);

fprintf('\nDone. Results saved to %s\n', results_dir);
exit;
