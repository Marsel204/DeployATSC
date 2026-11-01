"""Additive canonical-row populations; estimates never become observations."""


def identity_statistics(rows):
    rows = list(rows)
    return {'count': len(rows), 'confirmed': sum(r.get('red_wait_sec', 0.) for r in rows),
        'estimated': sum(r.get('estimated_red_wait_sec', 0.) for r in rows),
        'pending': sum(r.get('pending_estimated_red_wait_sec', 0.) for r in rows),
        'unresolved': sum(r.get('unresolved_coverage_sec', 0.) for r in rows),
        'uncertain_count': sum(bool(r.get('right_censored') or r.get('unknown_sec') or
            r.get('estimated_red_wait_sec') or r.get('quality_flags')) for r in rows)}


def identity_summary(stats):
    total = stats['confirmed']+stats['estimated']
    return {'identity_scheme': 'persistent_v1', 'estimated_red_wait_sec': round(stats['estimated'], 6),
        'pending_estimated_red_wait_sec': round(stats['pending'], 6),
        'best_estimate_red_wait_sec': round(total, 6),
        'best_estimate_mean_red_wait_sec': round(total/stats['count'], 6) if stats['count'] else None,
        'best_estimate_sample_count': stats['count'],
        'estimated_wait_fraction': stats['estimated']/total if total else None,
        'unresolved_coverage_sec': round(stats['unresolved'], 6),
        'unresolved_identity_count': stats['uncertain_count'],
        'identity_population_definition': 'Canonical logical visits, including partial/uncertain records. Unresolved splits may represent one physical vehicle; this count is not proof of exact physical uniqueness. Best estimates exclude pending provisional time.'}
