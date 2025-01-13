dataset_info = dict(
    dataset_name='cow2',
    paper_info=dict(
        author='dhx',
        title='cow: cow keypoint ',
        container='Proceedings of IEEE Conference on Computer '
        'Vision and Pattern Recognition (CVPR)',
        year='2019',
        homepage='https://github.com/Jeff-sjtu/CrowdPose',
    ),
    keypoint_info={
        0:
            dict(name='eye',
                 id=0, color=[51, 0, 255],
                 type='lower',
                 swap=''),
        1:
            dict(name='back',
                 id=1, color=[51, 153, 255],
                 type='upper',
                 swap=''),
        2:
            dict(
                name='shoulder',
                id=2,
                color=[51, 153, 255],
                type='upper',
                swap=''),
        3:
            dict(
                name='left_foreelbow',
                id=3,
                color=[51, 153, 255],
                type='upper',
                swap='right_foreelbow'),
        4:
            dict(
                name='left_forewrist',
                id=4,
                color=[51, 153, 255],
                type='upper',
                swap='right_forewrist'),
        5:
            dict(
                name='left_foreleg',
                id=5,
                color=[51, 153, 255],
                type='upper',
                swap='right_foreleg'),
        6:
            dict(
                name='right_foreelbow',
                id=6,
                color=[51, 153, 255],
                type='upper',
                swap='left_foreelbow'),
        7:
            dict(
                name='right_forewrist',
                id=7,
                color=[51, 153, 255],
                type='upper',
                swap='left_forewrist'),
        8:
            dict(
                name='right_foreleg',
                id=8,
                color=[51, 153, 255],
                type='upper',
                swap='left_foreleg'),
        9:
            dict(
                name='hip',
                id=9,
                color=[0, 255, 0],
                type='upper',
                swap=''),
        10:
            dict(
                name='buttock',
                id=10,
                color=[0, 255, 0],
                type='upper',
                swap=''),
        11:
            dict(
                name='right_ischium',
                id=11,
                color=[255, 128, 0],
                type='upper',
                swap=''),
        12:
            dict(
                name='left_hindleg',
                id=12,
                color=[0, 255, 0],
                type='upper',
                swap='right_hindleg'),
        13:
            dict(
                name='right_hindleg',
                id=13,
                color=[0, 255, 0],
                type='upper',
                swap='left_hindleg'),

        },
    skeleton_info={
        0:
            dict(link=('back', 'shoulder'), id=0, color=[0, 255, 0]),
        1:
            dict(link=('shoulder', 'left_foreelbow'), id=1, color=[255, 128, 0]),
        2:
            dict(link=('shoulder', 'right_foreelbow'), id=2, color=[51, 153, 255]),
        3:
            dict(link=('left_foreelbow', 'left_forewrist'), id=3, color=[255, 128, 0]),
        4:
            dict(link=('right_foreelbow', 'right_forewrist'), id=4, color=[51, 153, 255]),
        5:
            dict(link=('left_forewrist', 'left_foreleg'), id=5, color=[255, 128, 0]),
        6:
            dict(link=('right_forewrist', 'right_foreleg'), id=6, color=[51, 153, 255]),
        7:
            dict(link=('back', 'hip'), id=7, color=[51, 153, 255]),
        8:
            dict(link=('back', 'eye'), id=8, color=[51, 153, 255]),
        9:
            dict(link=('buttock', 'hip'), id=9, color=[51, 153, 255]),
        10:
            dict(link=('buttock', 'right_ischium'), id=10, color=[51, 153, 255]),
        11:
            dict(link=('right_ischium', 'left_hindleg'), id=11, color=[0, 255, 0]),
        12:
            dict(link=('right_ischium', 'right_hindleg'), id=12, color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1.5,
        1.5, 1.5, 1.5,
        1.5, 1.5, 1.5,
        1., 1., 1.,
        1.5, 1.5
    ],
# # yu xun lian
#     sigmas=[
#         0.02, 0.02, 0.02,
#         0.02, 0.02, 0.02,
#         0.02, 0.02, 0.02,
#         0.02, 0.02, 0.02,
#         0.02, 0.02,
#     ])
    # 85
    # sigmas=[
    #     0.01720, 0.03160, 0.02547,
    #     0.02792, 0.02621, 0.02370,
    #     0.02780, 0.02711, 0.02492,
    #     0.02293, 0.02351, 0.02825,
    #     0.02797, 0.2915,
    # ])
    # 80
    sigmas=[
        0.025490215762540243, 0.02826927162804384, 0.05677775091012153,
        0.026095593363104085, 0.0250332538888598, 0.05603581401651263,
        0.02495981999750956, 0.027735912684179503, 0.022273237026013113,
        0.030428390542162106, 0.027815943266800595, 0.025281019936829963,
        0.025926069767905204, 0.024238738664511122,
    ])