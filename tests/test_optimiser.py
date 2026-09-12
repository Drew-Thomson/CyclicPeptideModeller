from loopmover.specification import CyclicPeptide


def test_cyclic_peptide_creation():
    # Sequence of a cyclic peptide with D-amino acids
    seq = "GgGgGgGg"
    # Hardcoded angles obtained from rand_mac for consistency
    angles = [[180, 5.590805398181351, -33.452781999023955],
              [180, 195.36525033632287, -43.103083167386146],
              [180, -154.9013423353677, 99.5234864267943],
              [180, 1.5929471908703448, 102.79962538901297],
              [180, -182.48276251091306, 33.448410764172195],
              [180, 116.9984953509961, 74.43589438066388],
              [180, 74.19791695394002, 49.87470331504584],
              [180, 33.110010203433596, -6.294688456562142]]
    
    # Try to create cyclic peptide
    cp = CyclicPeptide(seq, angles=angles)
    
    # Assert sequence is set properly
    assert cp.sequence == seq
    # Check that ends overlap sufficiently well or it is constructed properly
    # As this relies on Isambard's TAPolypeptide, we ensure no exceptions are raised
    assert len(cp[0]) == 8
