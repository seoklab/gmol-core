from gmol.base.data.mmcif import Assembly, Mmcif


def test_save_load(sample_assembly: tuple[Mmcif, Assembly]):
    _, assembly = sample_assembly

    j1 = assembly.model_dump_json(indent=2)
    data = Assembly.model_validate_json(j1)

    j2 = data.model_dump_json(indent=2)

    assert j1 == j2
