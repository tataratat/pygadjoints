gismo_options = [
    {
        'tag': 'Function',
        'attributes': {'type': 'FunctionExpr', 'id': '1', 'dim': '2'},
        'text': '2*pi^2*sin(pi*x)*sin(pi*y)',
        'children': []
    },
    {'tag': 'boundaryConditions',
     'attributes': {'id': '2', 'multipatch': '0'},
     'text': '\n    ',
     'children': [
             {
                 'tag': 'Function',
                 'attributes': {
                     'type': 'FunctionExpr',
                     'dim': '2',
                     'index': '0'},
                 'text': 'sin(pi*x) * sin(pi*y)', 'children': []
             },
         {
                 'tag': 'Function',
                 'attributes': {
                     'type': 'FunctionExpr',
                             'dim': '2',
                             'index': '1',
                             'c': '2'
                 },
                 'text': '\n      ',
                 'children': [
                     {
                         'tag': 'c',
                         'attributes': {'index': '0'},
                         'text': 'pi*cos(pi*x) * sin(pi*y)',
                         'children': []
                     },
                     {
                         'tag': 'c',
                         'attributes': {'index': '1'},
                         'text': 'pi*sin(pi*x) * cos(pi*y)',
                         'children': []
                     }
                 ]
                 },
         {
                 'tag': 'Function',
                 'attributes': {
                     'type': 'FunctionExpr',
                     'dim': '2',
                     'index': '2'
                 },
                 'text': '0',
                 'children': []
                 },
         {
                 'tag': 'bc',
                 'attributes': {
                     'type': 'Dirichlet',
                     'function': '0',
                     'unknown': '0',
                                'name': 'dirichlet'
                 },
                 'text': '\n    ',
                 'children': []
                 },
         {'tag': 'bc',
                 'attributes': {
                     'unknown': '0',
                     'type': 'Neumann',
                             'function': '1',
                             'name': 'neumann'
                 },
                 'text': '\n    ',
                 'children': []
          }
     ]
     },
    {'tag': 'Function',
     'attributes': {'type': 'FunctionExpr', 'id': '3', 'dim': '2'},
     'text': 'sin(pi*x) * sin(pi*y)', 'children': []
     },
    {
        'tag': 'OptionList',
        'attributes': {'id': '4'},
        'text': '\n  ',
        'children': [
            {
                'tag': 'int',
                'attributes':
                {'label': 'DirichletStrategy',
                          'desc': 'Method for enforcement of Dirichlet BCs [11..14]',
                          'value': '11'},
                'text': None,
                'children': []
            },
            {
                'tag': 'int',
                'attributes': {
                    'label': 'DirichletValues',
                             'desc': 'Method for computation of Dirichlet DoF values [100..103]',
                             'value': '101'
                },
                'text': None,
                'children': []
            },
            {
                'tag': 'int',
                'attributes': {
                    'label': 'InterfaceStrategy',
                             'desc': 'Method of treatment of patch interfaces [0..3]',
                             'value': '1'
                },
                'text': None,
                'children': []
            },
            {'tag': 'real',
             'attributes': {
                 'label': 'bdA',
                 'desc': 'Estimated nonzeros per column of the matrix: bdA*deg + bdB',
                 'value': '2'
             },
             'text': None,
             'children': []
             },
            {'tag': 'int',
             'attributes': {
                 'label': 'bdB',
                 'desc': 'Estimated nonzeros per column of the matrix: bdA*deg + bdB',
                 'value': '1'
             },
             'text': None,
             'children': []
             },
            {
                'tag': 'real',
                'attributes': {
                    'label': 'bdO',
                             'desc': 'Overhead of sparse mem. allocation: (1+bdO)(bdA*deg + bdB) [0..1]',
                             'value': '0.333'},
                'text': None,
                'children': []
            },
            {'tag': 'real',
             'attributes': {
                 'label': 'quA',
                 'desc': 'Number of quadrature points: quA*deg + quB',
                 'value': '1'
             },
             'text': None,
             'children': []
             },
            {'tag': 'int',
             'attributes': {
                 'label': 'quB',
                 'desc': 'Number of quadrature points: quA*deg + quB',
                 'value': '1'
             },
             'text': None,
             'children': []
             },
            {
                'tag': 'int',
                'attributes': {
                    'label': 'quRule',
                             'desc': 'Quadrature rule [1:GaussLegendre,2:GaussLobatto]',
                             'value': '1'},
                'text': None, 'children': []
            }
        ]
    }
]
