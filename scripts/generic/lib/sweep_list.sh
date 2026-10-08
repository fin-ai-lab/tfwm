#!/bin/bash
# sweep_list.sh — resolve the sweep files a submitter was handed into the env
# contract slurm_train_bundle.sh takes. Sourced by run_all_months_sweep.sh.
#
# WHY A LIST AND NOT ONE FILE — i.e. why month-major is a property of the JOB.
#
# Everything a job pays for besides the training itself is keyed on the MONTH:
# the mosaic rsync, the shard materialization, the anchor tables, the ~26 GB
# pre-built eval panel. All of it is NODE-LOCAL. The only way to make several
# sweeps share that work is to put them in the SAME JOB. Submitting them as
# separate jobs with adjacent job IDs (what submit_month_major.sh used to do)
# biases only the ORDER slurm starts them in -- with many GPUs they start on
# different nodes and each one stages the month again from scratch.
#
# Requires from the caller: LOCAL_REPO and info()/warn()/error() (either
# tree's lib/common.sh provides them).
#
# resolve_sweep_list <sweep file>... sets:
#   SWEEP_FILES_REL  space-separated repo-relative paths (the bundle's env)
#   SWEEP_NAMES      SWEEP_NAME per file, in order
#   SWEEP_NVALUES    value count per file, in order
#   SWEEP_TOTAL      values one month's job will run back to back
#   SWEEP_JOB_LABEL  sbatch --job-name prefix
#   SBATCH_EXTRA     every sweep's own flags, then the caller's (last wins)

# Print "NAME<TAB>NVALUES<TAB>SBATCH_EXTRA" for one sweep file.
#
# In a SUBSHELL, deliberately: sourcing several sweep files into the
# submitter's own shell would let one file's SWEEP_VALUES (an array — a
# shorter sweep inherits the tail of a longer one) and hook functions leak
# into the next. Anything a sweep file prints goes to stderr so it cannot
# corrupt the record. SBATCH_EXTRA starts empty so we capture only what THIS
# file adds; the caller's own SBATCH_EXTRA is appended once, at the end.
_sweep_meta() {
    (
        set -euo pipefail
        SBATCH_EXTRA=""
        # shellcheck disable=SC1090
        source "$1" >&2
        printf '%s\t%s\t%s\t%s\n' \
            "${SWEEP_NAME:?SWEEP_NAME not set}" \
            "${#SWEEP_VALUES[@]}" \
            "${SWEEP_MONTH_SET:-any}" \
            "${SBATCH_EXTRA}"
    )
}

resolve_sweep_list() {
    [ $# -gt 0 ] || error "No sweep file given."

    SWEEP_FILES_REL=""
    SWEEP_NAMES=()
    SWEEP_NVALUES=()
    SWEEP_TOTAL=0

    local sweep_extra="" f abs rel meta name nvals month_set extra
    SWEEP_MONTH_SETS=()
    for f in "$@"; do
        abs="${f}"
        [[ "${abs}" = /* ]] || abs="${LOCAL_REPO}/scripts/${abs}"
        [ -f "${abs}" ] || error "Sweep file not found: ${abs}"
        rel="$(realpath --relative-to="${LOCAL_REPO}" "${abs}")"
        case "${rel}" in
            *[[:space:]]*) error "Sweep path contains whitespace (SWEEP_FILES_REL is space-separated): ${rel}" ;;
        esac

        meta="$(_sweep_meta "${abs}")" \
            || error "Sweep file failed to load (SWEEP_NAME/SWEEP_VALUES missing, or a syntax error): ${rel}"
        IFS=$'\t' read -r name nvals month_set extra <<< "${meta}"
        [ "${nvals}" -gt 0 ] || error "${rel}: SWEEP_VALUES is empty"

        SWEEP_FILES_REL="${SWEEP_FILES_REL:+${SWEEP_FILES_REL} }${rel}"
        SWEEP_NAMES+=("${name}")
        SWEEP_NVALUES+=("${nvals}")
        SWEEP_MONTH_SETS+=("${month_set}")
        SWEEP_TOTAL=$(( SWEEP_TOTAL + nvals ))
        if [ -n "${extra}" ]; then
            sweep_extra="${sweep_extra:+${sweep_extra} }${extra}"
        fi
    done

    if [ ${#SWEEP_NAMES[@]} -eq 1 ]; then
        SWEEP_JOB_LABEL="${SWEEP_NAMES[0]}"
    else
        SWEEP_JOB_LABEL="${SWEEP_NAMES[0]}+$(( ${#SWEEP_NAMES[@]} - 1 ))"
    fi

    # A sweep file's own flags first, the caller's env SBATCH_EXTRA last:
    # sbatch takes the LAST of a repeated option, which is what
    # SBATCH_EXTRA="--nice=0" relies on to beat a sweep's --nice=10000.
    SBATCH_EXTRA="${sweep_extra}${SBATCH_EXTRA:+ ${SBATCH_EXTRA}}"

    local i
    for i in "${!SWEEP_NAMES[@]}"; do
        info "  sweep $((i + 1))/${#SWEEP_NAMES[@]}: ${SWEEP_NAMES[i]} (${SWEEP_NVALUES[i]} values) — $(echo "${SWEEP_FILES_REL}" | cut -d' ' -f$((i + 1)))"
    done

    # The grouped job's whole point is one node per month, which means the
    # sweeps run BACK TO BACK on that node's single GPU. Wall clock is the sum.
    if [ ${#SWEEP_NAMES[@]} -gt 1 ]; then
        warn "Grouped job: ${#SWEEP_NAMES[@]} sweeps x ${SWEEP_TOTAL} values per month, back to back on ONE GPU."
        case "${SBATCH_EXTRA}" in
            *--time=*) warn "  A --time= limit is in effect; it must cover the SUM of all ${#SWEEP_NAMES[@]} sweeps: ${SBATCH_EXTRA}" ;;
            *)         warn "  Wall clock is the SUM of the sweeps; raise it with SBATCH_EXTRA=\"--time=...\" if the default is short." ;;
        esac
    fi
}


# WHICH MONTHS A SWEEP IS FOR, ENFORCED (2026-09-13).
#
# The month set is a property of the EXPERIMENT, and until now it lived only in
# the launcher you happened to type. Nothing in a sweep file said which set it
# belonged to and no launcher checked, so supervised_specialists.sh -- the
# 32-month reported panel -- was submitted through run_full_data_supervised.sh
# and ran 198 months of full history instead. It trained for hours before
# anyone noticed, because a wrong month set is not an error, just a different
# and much larger experiment.
#
# So a sweep declares SWEEP_MONTH_SET and its launcher declares what it serves:
#
#   sampled32      the 32 reported sweep months (run_all_months_sweep.sh)
#   full_history   every month the mosaic has  (specific/run_full_data_supervised.sh)
#   any            the sweep does not care; the launcher decides
#
# A sweep that declares nothing is "any" and still runs anywhere, so this is
# additive: only a sweep that states a set can be refused.
require_month_set() {
    local provided="$1" i bad=0
    for i in "${!SWEEP_MONTH_SETS[@]}"; do
        local want="${SWEEP_MONTH_SETS[i]}"
        [ "${want}" = "any" ] && continue
        [ "${want}" = "${provided}" ] && continue
        echo "ERROR: $(echo "${SWEEP_FILES_REL}" | cut -d' ' -f$((i + 1))) declares" >&2
        echo "       SWEEP_MONTH_SET=${want}, but this launcher submits '${provided}'." >&2
        bad=1
    done
    [ "${bad}" = "0" ] || error "sweep/launcher month-set mismatch — see above for the launcher that serves it"
}
