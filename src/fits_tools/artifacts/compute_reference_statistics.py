





















def compute_reference_statistics(
    frame_paths: List[Union[str, Path]],
    ext: int = 0,
    existing_bpm: Optional[np.ndarray] = None,
    memmap: bool = True,
    streaming: bool = True,
    chunk_size: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel reference statistics (auto-selects method).
    
    Automatically chooses between streaming (memory-efficient) and
    chunked (supports masking) modes based on your requirements.
    
    Parameters
    ----------
    ... [existing parameters] ...
    streaming : bool, optional
        Use streaming algorithms (constant memory).
        - True: P² algorithm, constant memory, no mask support
        - False: Load in chunks, higher memory, supports masking
        - 'auto': Choose automatically based on dataset size and mask
        Default: True.
    
    Notes
    -----
    **Auto-selection logic** (``streaming='auto'``):
    
    - If existing_bpm provided: Use chunked (masking required)
    - If N_frames > 50: Use streaming (memory critical)
    - Otherwise: Use chunked (best accuracy)
    """
    
    # Auto-select method
    if streaming == 'auto':
        if existing_bpm is not None:
            streaming = False  # Need masking
            if verbose:
                logger.info("Auto-selected chunked mode (masking required)")
        elif len(frame_paths) > 50:
            streaming = True  # Memory critical
            if verbose:
                logger.info("Auto-selected streaming mode (large dataset)")
        else:
            streaming = False  # Best accuracy for small dataset
            if verbose:
                logger.info("Auto-selected chunked mode (small dataset)")
    
    if streaming:
        if existing_bpm is not None:
            logger.warning(
                "Streaming mode does not support masking. "
                "Bad pixels will be included. Set streaming=False to use masking."
            )
        return compute_reference_statistics_streaming(
            frame_paths, ext=ext, memmap=memmap, verbose=verbose
        )
    else:
        return compute_reference_statistics_chunked(
            frame_paths, ext=ext, existing_bpm=existing_bpm,
            chunk_size=chunk_size, verbose=verbose
        )