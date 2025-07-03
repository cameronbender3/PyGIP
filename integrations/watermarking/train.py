import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import calculate_metrics 
from .trigger import TriggerGenerator, generate_trigger_graph


def train_model(model, data, epochs=200, wm_weight=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)

        main_loss = criterion(out[data.train_mask], data.y[data.train_mask])

        wm_loss = 0
        if hasattr(data, 'trigger_nodes'):
            trigger_mask = data.trigger_mask
            wm_loss = criterion(out[trigger_mask], data.y[trigger_mask])

        loss = (1-wm_weight)*main_loss + wm_weight*wm_loss
        loss.backward()
        optimizer.step()

    return calculate_metrics(model, data)




def bi_level_optimization(target_model, generator, data, epochs=100, inner_steps=5):
    optimizer_model = torch.optim.Adam(target_model.parameters(), lr=0.01)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for _ in range(inner_steps):
            optimizer_model.zero_grad()
            trigger_data = generate_trigger_graph(data, generator, target_model)

            out_clean = target_model(data.x, data.edge_index)
            out_trigger = target_model(trigger_data.x, trigger_data.edge_index)

            clean_loss = criterion(out_clean[data.train_mask], data.y[data.train_mask])
            trigger_loss = criterion(out_trigger[trigger_data.trigger_mask],
                                  trigger_data.y[trigger_data.trigger_mask])

            total_loss = clean_loss + trigger_loss
            total_loss.backward()
            optimizer_model.step()

        optimizer_gen.zero_grad()
        trigger_data = generate_trigger_graph(data, generator, target_model)

        orig_features = data.x[trigger_data.selected_nodes]
        trigger_features = trigger_data.x[trigger_data.trigger_nodes]
        sim_loss = -F.cosine_similarity(orig_features.unsqueeze(1),
                                     trigger_features.unsqueeze(0), dim=-1).mean()

        out = target_model(trigger_data.x, trigger_data.edge_index)
        trigger_loss = criterion(out[trigger_data.trigger_mask],
                               trigger_data.y[trigger_data.trigger_mask])

        owner_loss = F.binary_cross_entropy(
            trigger_data.x[trigger_data.trigger_nodes, -5:],
            generator.owner_id.expand(len(trigger_data.trigger_nodes), 5)
        )

        total_gen_loss = 0.4*sim_loss + 0.4*trigger_loss + 0.2*owner_loss
        total_gen_loss.backward()
        optimizer_gen.step()
