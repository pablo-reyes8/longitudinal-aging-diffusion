"""Explicitly configurable name-based test/reference identity check."""
import warnings


def check_identity_overlap(config, panel, reference):
    overlap = sorted(set(panel.identity_id.astype(str)) &
                     set(map(str, reference.metadata.get('train_identities', []))))
    enabled = getattr(config, 'check_identity_overlap', False)
    if overlap and enabled:
        raise ValueError(f'Test/reference training identity overlap: {overlap[:10]}')
    if overlap and not enabled:
        warnings.warn('Identity-name overlap check explicitly bypassed; matching names '
                      'are not verified distinct people.', UserWarning, stacklevel=2)
    return dict(enabled=enabled, overlapping_names=overlap,
                status='bypassed' if not enabled else 'checked')
