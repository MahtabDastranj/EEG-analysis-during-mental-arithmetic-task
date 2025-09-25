%% Batch export EEG data + ICA (EEGLAB)
% Inputs
input_dir  = "E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\Processed data\eeglab files";
output_dir = "E:\AUT\thesis\EEG-analysis-during-mental-arithmetic-task\Processed data\exported";

% Optional: ensure EEGLAB is on path (uncomment and set to your EEGLAB folder)
% addpath("C:\eeglab2025"); eeglab; close;  % opens then closes GUI, keeps paths

if ~isfolder(output_dir), mkdir(output_dir); end

% Find all .set files
files = dir(fullfile(input_dir, "*.set"));
if isempty(files)
    error("No .set files found in: %s", input_dir);
end

for k = 1:numel(files)
    try
        in_path  = fullfile(files(k).folder, files(k).name);
        [~, base, ~] = fileparts(files(k).name);
        out_sub  = fullfile(output_dir, base);
        if ~isfolder(out_sub), mkdir(out_sub); end

        fprintf('\n=== Processing %s ===\n', files(k).name);

        % Load with EEGLAB (handles paired .fdt automatically)
        EEG = pop_loadset('filename', files(k).name, 'filepath', files(k).folder);

        % Basic info
        nbchan = EEG.nbchan;
        pnts   = EEG.pnts;
        fs     = EEG.srate;

        % Data is channels x time in EEGLAB
        data = double(EEG.data);               % (nbchan x pnts)

        % --- Export raw data: time x channels (to match typical analytics pipelines)
        writematrix(data.', fullfile(out_sub, "data.txt"), 'Delimiter','\t');

        % --- Fetch ICA pieces
        W = []; S = []; A = [];
        if isfield(EEG, 'icaweights'), W = double(EEG.icaweights); end       % (nComp x nbchan)
        if isfield(EEG, 'icasphere'),  S = double(EEG.icasphere);  end       % (nbchan x nbchan)
        if isfield(EEG, 'icawinv'),    A = double(EEG.icawinv);    end       % (nbchan x nComp)

        if ~isempty(W)
            % If sphere missing/empty, use identity
            if isempty(S)
                S = eye(size(W,2));
                warning('icasphere missing/empty in %s — using identity.', files(k).name);
            end

            % Sanity checks / common shape fixes
            if size(W,2) ~= nbchan && size(W,1) == nbchan
                W = W.'; % transpose if saved as (nbchan x nComp)
            end
            if size(W,2) ~= nbchan
                error('icaweights shape %s incompatible with data channels %d in %s.', ...
                      mat2str(size(W)), nbchan, files(k).name);
            end
            if ~isequal(size(S), [nbchan nbchan])
                if isequal(size(S.'), [nbchan nbchan])
                    S = S.'; % fix transpose
                else
                    error('icasphere shape %s incompatible with channels %d in %s.', ...
                          mat2str(size(S)), nbchan, files(k).name);
                end
            end

            % Activations: (nComp x pnts)
            % EEGLAB convention: activations = W * (S * data)
            ica_act = W * (S * data);

            % If mixing missing/wrong, derive from WS
            if isempty(A) || ~isequal(size(A), [nbchan size(W,1)])
                A = pinv(W * S);  % (nbchan x nComp)
            end

            % --- Save ICA exports
            writematrix(W,            fullfile(out_sub, "icaweights.txt"),  'Delimiter','\t');  % comp x ch
            writematrix(A,            fullfile(out_sub, "icawinv.txt"),     'Delimiter','\t');  % ch x comp
            writematrix(S,            fullfile(out_sub, "icasphere.txt"),   'Delimiter','\t');  % ch x ch
            writematrix(ica_act.',    fullfile(out_sub, "ica_activity.txt"),'Delimiter','\t');  % time x comp

            fprintf('  ✓ Exported: data(%dx%d), W(%dx%d), S(%dx%d), A(%dx%d), act(%dx%d)\n', ...
                nbchan, pnts, size(W,1), size(W,2), size(S,1), size(S,2), size(A,1), size(A,2), size(ica_act,1), size(ica_act,2));
        else
            fprintf('  ⚠ No ICA found in %s (icaweights empty). Exported data only.\n', files(k).name);
        end

        % (Optional) write a small meta file for auditing
        meta = struct( ...
            'file', files(k).name, ...
            'setname', getfield(EEG, 'setname', ''), ... %#ok<GFLD>
            'sfreq', fs, ...
            'n_channels', nbchan, ...
            'n_times', pnts, ...
            'has_ica', ~isempty(W), ...
            'shapes', struct( ...
                'data', [nbchan pnts], ...
                'icaweights', size(W), ...
                'icasphere', size(S), ...
                'icawinv', size(A) ...
            ) ...
        );
        fid = fopen(fullfile(out_sub, "meta.txt"), 'w');
        fprintf(fid, 'file: %s\nsetname: %s\nsfreq: %g\nn_channels: %d\nn_times: %d\nhas_ica: %d\n', ...
            meta.file, meta.setname, meta.sfreq, meta.n_channels, meta.n_times, meta.has_ica);
        fprintf(fid, 'shapes.data = [%d %d]\n', meta.shapes.data(1), meta.shapes.data(2));
        fprintf(fid, 'shapes.icaweights = [%d %d]\n', size(W,1), size(W,2));
        fprintf(fid, 'shapes.icasphere  = [%d %d]\n', size(S,1), size(S,2));
        fprintf(fid, 'shapes.icawinv    = [%d %d]\n', size(A,1), size(A,2));
        fclose(fid);

    catch ME
        warning('ERROR on %s: %s', files(k).name, ME.message);
    end
end

fprintf('\nDone. Exports are in: %s\n', output_dir);
