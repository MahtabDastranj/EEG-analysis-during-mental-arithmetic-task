# ========================================================================
# Batch export of EEG data + ICA activity, and ICA weights from EEGLAB
# ========================================================================
# This script:
#   1. Loops through all .set files in a folder
#   2. Exports EEG data and ICA activations to a text file
#   3. Exports ICA weight matrix to a text file
#
# Notes:
# - 'transpose','on' flips the matrix so that:
#     * For data/ICA activity: rows = timepoints, columns = channels/components
#       (easier to read in Python, samples along rows)
#     * For ICA weights: rows = components, columns = channels
#       (standard EEGLAB ICA weight layout)
# ========================================================================

# ----------- User settings -----------
input_dir = 'C:\path\to\your\set\files'   # folder with .set files
output_dir = 'C:\path\to\output\txt'      # folder for txt exports
# -------------------------------------

if ~exist(output_dir, 'dir'):
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, '*.set'));

[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab; # start EEGLAB

for i = 1:length(files)
    fname = files(i).name;
    fprintf('Processing %s...\n', fname);

    # Load dataset
    EEG = pop_loadset('filename', fname, 'filepath', input_dir);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);

    # 1. Export data and ICA activity
    # 'transpose','on' ensures:
    #   Rows = timepoints, Columns = channels + ICA components
    out_data_file = fullfile(output_dir, [fname(1:end-4) '_dataICA.txt']);
    pop_export(EEG, out_data_file, 'transpose', 'on', 'ica', 'on');

    # 2. Export ICA weight matrix
    # 'transpose','on' ensures:
    #   Rows = components, Columns = channels
    out_weights_file = fullfile(output_dir, [fname(1:end-4) '_weights.txt']);
    pop_export(EEG, out_weights_file, 'transpose', 'on', 'icaweights', 'on');
end

fprintf('Done! All files exported to %s\n', output_dir);
