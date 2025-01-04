model = dict(
    main=dict(
        type='MyGIN',
        num_node_emb_list=[39],
        num_edge_emb_list=[10],
        num_layers=4,
        emb_dim=128,
        readout='mean',
        JK='last',
        dropout=0.1,
    )
)