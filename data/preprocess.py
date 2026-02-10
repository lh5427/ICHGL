import pandas as pd
import numpy as np
import json



def list2set(l):
    return set([tuple(edge) for edge in l])

def list2dict(l, n_users):
    user_interaction_dict = dict()
    for edge in l:
        user, item = int(edge[0]), int(edge[1])
        if user not in user_interaction_dict.keys():
            user_interaction_dict[user] = []
        user_interaction_dict[user].append(item)
    return user_interaction_dict

def preprocess():
    print("🔥 preprocess started")

    # for data in ['taobao', 'tmall', 'jdata', 'beibei', 'taobao1', 'tmall1', 'ML10M1', 'Yelp1', 'tenrec']:
    for data in ['tenrec']:
        print(f'Preprocessing {data}...')
        
        # Load behavior-specific graph
        view = np.loadtxt(f'data/{data}/view.txt', dtype=int)
        cart = np.loadtxt(f'data/{data}/cart.txt', dtype=int)
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            collect = np.loadtxt(f'data/{data}/collect.txt', dtype=int)
        train_buy = np.loadtxt(f'data/{data}/train.txt', dtype=int)
        test_buy = np.loadtxt(f'data/{data}/test.txt', dtype=int)

        # ===================== PATCH 1 ==============================
        # 构造集合，便于后续 only / overlap 计算
        # ============================================================

        view_set = list2set(view)
        cart_set = list2set(cart)
        buy_set = list2set(train_buy)

        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            collect_set = list2set(collect)
        else:
            collect_set = set()
        # ============================================================


        # Preprocess target-complemented behaviors
        view_not_buy = list2set(view).difference(list2set(train_buy))
        cart_not_buy = list2set(cart).difference(list2set(train_buy))
        # buy.txt 与 view.txt 的补集（存在于buy但不存在于view）
        buy_not_view = list2set(train_buy).difference(list2set(view))
        # buy.txt 与 collect.txt 的补集（存在于buy但不存在于cart）
        buy_not_cart = list2set(train_buy).difference(list2set(cart))
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            collect_not_buy = list2set(collect).difference(list2set(train_buy))
            buy_not_collect = list2set(train_buy).difference(list2set(collect))
        
        np.savetxt(f'data/{data}/view_not_buy.txt', sorted(list(view_not_buy)), fmt='%d')
        np.savetxt(f'data/{data}/cart_not_buy.txt', sorted(list(cart_not_buy)), fmt='%d')
        np.savetxt(f'data/{data}/buy_not_view.txt', sorted(list(buy_not_view)), fmt='%d')
        np.savetxt(f'data/{data}/buy_not_cart.txt', sorted(list(buy_not_cart)), fmt='%d')
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            np.savetxt(f'data/{data}/collect_not_buy.txt', sorted(list(collect_not_buy)), fmt='%d')
            np.savetxt(f'data/{data}/buy_not_collect.txt', sorted(list(buy_not_collect)), fmt='%d')

        # Preprocess target-intersected behaviors
        view_buy = list2set(view).intersection(list2set(train_buy))
        cart_buy = list2set(cart).intersection(list2set(train_buy))
        view_cart = list2set(view).intersection(list2set(cart))
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            collect_buy = list2set(collect).intersection(list2set(train_buy))
            view_collect = list2set(view).intersection(list2set(collect))
            cart_collect = list2set(cart).intersection(list2set(collect))
        
        np.savetxt(f'data/{data}/view_buy.txt', sorted(list(view_buy)), fmt='%d')
        np.savetxt(f'data/{data}/cart_buy.txt', sorted(list(cart_buy)), fmt='%d')
        np.savetxt(f'data/{data}/view_cart.txt', sorted(list(view_cart)), fmt='%d')
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            np.savetxt(f'data/{data}/collect_buy.txt', sorted(list(collect_buy)), fmt='%d')
            np.savetxt(f'data/{data}/view_collect.txt', sorted(list(view_collect)), fmt='%d')
            np.savetxt(f'data/{data}/cart_collect.txt', sorted(list(cart_collect)), fmt='%d')
        

        assert len(view) == len(view_not_buy) + len(view_buy)
        assert len(cart) == len(cart_not_buy) + len(cart_buy)
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            assert len(collect) == len(collect_not_buy) + len(collect_buy)


        # ===================== PATCH 2 ==============================
        # 行为类型1：Only-BSG
        # ============================================================

        view_only = view_set.difference(cart_set).difference(collect_set).difference(buy_set)
        cart_only = cart_set.difference(view_set).difference(collect_set).difference(buy_set)
        # 添加 buy_only 的计算
        buy_only = buy_set.difference(view_set).difference(cart_set).difference(collect_set)

        np.savetxt(f'data/{data}/view_only.txt', sorted(list(view_only)), fmt='%d')
        np.savetxt(f'data/{data}/cart_only.txt', sorted(list(cart_only)), fmt='%d')
        np.savetxt(f'data/{data}/buy_only.txt', sorted(list(buy_only)), fmt='%d')  # 新增

        if collect_set:
            collect_only = collect_set.difference(view_set).difference(cart_set).difference(buy_set)
            np.savetxt(f'data/{data}/collect_only.txt', sorted(list(collect_only)), fmt='%d')
        # ============================================================

        # bsg_only 并集（view_only + cart_only + collect_only）
        # ================================================================

        bsg_only_union = set()
        bsg_only_union |= view_only
        bsg_only_union |= cart_only
        if collect_set:
            bsg_only_union |= collect_only
        # ================================================================

        # ===================== PATCH 3 ==============================
        # 行为类型2：Overlap-BSG
        # （补集：bsg − bsg_only）
        # ============================================================

        view_overlap = view_set.difference(view_only)
        cart_overlap = cart_set.difference(cart_only)

        np.savetxt(f'data/{data}/view_overlap.txt', sorted(list(view_overlap)), fmt='%d')
        np.savetxt(f'data/{data}/cart_overlap.txt', sorted(list(cart_overlap)), fmt='%d')

        if collect_set:
            collect_overlap = collect_set.difference(collect_only)
            np.savetxt(f'data/{data}/collect_overlap.txt', sorted(list(collect_overlap)), fmt='%d')
        # ============================================================


        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 新增：生成最后一层多行为交集和补集文件
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if data == 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            # view.txt 与 collect.txt 与 buy.txt 的交集
            view_cart_buy = list2set(view).intersection(list2set(cart)).intersection(list2set(train_buy))
            np.savetxt(f'data/{data}/view_cart_buy.txt', sorted(list(view_cart_buy)), fmt='%d')

            # buy.txt 与 view.txt 与 collect.txt 的补集（存在于buy但不存在于view且不存在于cart）
            buy_not_view_not_cart = list2set(train_buy).difference(list2set(view)).difference(list2set(cart))
            np.savetxt(f'data/{data}/buy_not_view_not_cart.txt', sorted(list(buy_not_view_not_cart)), fmt='%d')
        else:
            # view.txt 与 collect.txt 与 cart.txt 与 buy.txt 的交集
            view_cart_collect_buy = list2set(view).intersection(list2set(cart)).intersection(
                list2set(collect)).intersection(list2set(train_buy))
            np.savetxt(f'data/{data}/view_cart_collect_buy.txt', sorted(list(view_cart_collect_buy)), fmt='%d')

            # buy.txt 与 view.txt 与 collect.txt 与 cart.txt 的补集
            buy_not_view_not_cart_not_collect = list2set(train_buy).difference(list2set(view)).difference(
                list2set(cart)).difference(list2set(collect))
            np.savetxt(f'data/{data}/buy_not_view_not_cart_not_collect.txt',
                       sorted(list(buy_not_view_not_cart_not_collect)), fmt='%d')
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 新增：生成 view_and_buy.txt、cart_and_buy.txt 和 collect_and_buy.txt
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        view_and_buy = list2set(view).union(list2set(train_buy))
        cart_and_buy = list2set(cart).union(list2set(train_buy))
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            collect_and_buy = list2set(collect).union(list2set(train_buy))

        np.savetxt(f'data/{data}/view_and_buy.txt', sorted(list(view_and_buy)), fmt='%d')
        np.savetxt(f'data/{data}/cart_and_buy.txt', sorted(list(cart_and_buy)), fmt='%d')

        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            np.savetxt(f'data/{data}/collect_and_buy.txt', sorted(list(collect_and_buy)), fmt='%d')
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


        # Preprocess unified behavior graph
        all_edge = list2set(view).union(list2set(cart)).union(list2set(train_buy))
        if data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1':
            all_edge = all_edge.union(list2set(collect))
        all_edge = np.array(sorted(list(all_edge)))
        np.savetxt(f'data/{data}/ubg.txt', all_edge, fmt='%d')

        # 新行为类型：ubg - bsg_only
        # ================================================================

        ubg_minus_bsg_only = list2set(all_edge).difference(bsg_only_union)

        np.savetxt(
            f'data/{data}/ubg_minus_bsg_only.txt',
            np.array(sorted(list(ubg_minus_bsg_only))),
            fmt='%d'
        )
        # ================================================================
        # 新增：buy_minus_buy_only (buy行为的所有交互 - buy_only类型的交互)
        # ================================================================

        buy_minus_buy_only = buy_set.difference(buy_only)

        np.savetxt(
            f'data/{data}/buy_minus_buy_only.txt',
            np.array(sorted(list(buy_minus_buy_only))),
            fmt='%d'
        )
        # ================================================================
        
        
        # Generate train/test buy interaction dict
        n_users = all_edge[:,0].max()
        n_items = all_edge[:,1].max()
            
        train_dict = list2dict(train_buy, n_users)
        test_dict = list2dict(test_buy, n_items)
        
        with open(f'data/{data}/train.json', 'w') as f:
            json.dump(train_dict, f)
        with open(f'data/{data}/test.json', 'w') as f:
            json.dump(test_dict, f)
        
        
        # Generate data statistics
        bsg_types = ['view', 'cart', 'buy'] if (data == 'taobao' or data == 'beibei' or data == 'taobao1' or data == 'ML10M1') else ['view', 'cart', 'collect', 'buy']
        tcb_types = ['view_not_buy', 'cart_not_buy'] if (data == 'taobao' or data == 'beibei' or data == 'taobao1' or data == 'ML10M1') else ['view_not_buy', 'collect_not_buy', 'cart_not_buy']
        tib_types = ['view_buy', 'cart_buy'] if (data == 'taobao' or data == 'beibei' or data == 'taobao1' or data == 'ML10M1') else ['view_buy', 'collect_buy', 'cart_buy']
        trbg_types = tcb_types + tib_types
        
        data_statistics = {
            'n_users': int(n_users),
            'n_items': int(n_items),
            'bsg_types': bsg_types,
            'tcb_types': tcb_types,
            'tib_types': tib_types,
            'trbg_types': trbg_types,
            # ===================== PATCH 5 ==============================
            'bsg_only_types': ['view_only', 'cart_only'] + (['collect_only'] if collect_set else [])+ ['buy_only'],
            'bsg_overlap_types': ['view_overlap', 'cart_overlap'] + (['collect_overlap'] if collect_set else []),
            'buy_minus_buy_only_types': ['buy_minus_buy_only'],
            # ============================================================
            # ===================== PATCH NEW-3 ==============================
            'ubg_minus_bsg_only_types': ['ubg_minus_bsg_only'],
            'n_ubg_minus_bsg_only': len(ubg_minus_bsg_only),
            # ================================================================
            'n_view': len(view),
            'n_cart': len(cart),
            'n_collect': len(collect) if (data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1') else 0,
            'n_buy': len(train_buy),
            'n_view_not_buy': len(view_not_buy),
            'n_cart_not_buy': len(cart_not_buy),
            'n_collect_not_buy': len(collect_not_buy) if (data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1') else 0,
            'n_view_buy': len(view_buy),
            'n_cart_buy': len(cart_buy),
            'n_collect_buy': len(collect_buy) if (data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1') else 0,
            'n_ubg': len(all_edge),
            'n_view_and_buy': len(view_and_buy),  # 新增
            'n_cart_and_buy': len(cart_and_buy),  # 新增
            'n_collect_and_buy': len(collect_and_buy) if (data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1') else 0,  # 新增
            'n_view_cart': len(view_cart),  # 新增
            'n_view_collect': len(view_collect) if (data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1') else 0,  # 新增
            'n_cart_collect': len(cart_collect) if (data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1') else 0,  # 新增
            'n_buy_not_view': len(buy_not_view),  # 新增
            'n_buy_not_cart': len(buy_not_cart),  # 新增
            'n_buy_not_collect': len(buy_not_collect) if (data != 'taobao' and data != 'beibei' and data != 'taobao1' and data != 'ML10M1') else 0,  # 新增
            'n_view_cart_buy': len(view_cart_buy) if data == 'taobao' else 0,  # 新增
            'n_buy_not_view_not_cart': len(buy_not_view_not_cart) if data == 'taobao' else 0,  # 新增
            'n_view_cart_collect_buy': len(view_cart_collect_buy) if (data != 'taobao' and data != 'beibei') else 0,  # 新增
            'n_buy_not_view_not_cart_not_collect': len(buy_not_view_not_cart_not_collect) if (data != 'taobao' and data != 'beibei') else 0,  # 新增
            # ===================== PATCH 4 ==============================
            'n_view_only': len(view_only),
            'n_cart_only': len(cart_only),
            'n_collect_only': len(collect_only) if collect_set else 0,
            'n_buy_only': len(buy_only),
            'n_buy_minus_buy_only': len(buy_minus_buy_only),

            'n_view_overlap': len(view_overlap),
            'n_cart_overlap': len(cart_overlap),
            'n_collect_overlap': len(collect_overlap) if collect_set else 0,
            # ============================================================

        }
        with open(f'data/{data}/statistics.json', 'w') as f:
            json.dump(data_statistics, f)
                

if __name__ == '__main__':
    preprocess()