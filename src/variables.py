"""Variable dictionary (deliverable L1): variable -> signification -> type -> rôle -> justification."""
import pandas as pd

from src.config import RESULTS_DIR, ROOT

E, C, I, X = "entrée", "cible", "identifiant", "à exclure"

VARIABLES = [
    # ---------------- identifiers ----------------
    ("scenario_id", "flows", "Numéro du scénario (graph_K / routing_K / slices_K)", "entier", I,
     "Unité de découpage train/val/test (split groupé) ; jamais une feature."),
    ("flow_id", "flows", "Index du flux dans son scénario", "entier", I, "Clé technique, aucune signification physique."),
    ("topology", "scenarios", "Nom de la topologie Topology Zoo (162 distinctes)", "catégoriel", I,
     "Sert au split « topologie inédite » ; l'utiliser comme feature empêcherait de généraliser à une topologie nouvelle."),
    ("src, dst, antenna", "flows", "Nœuds origine, destination et antenne du flux", "entier", I,
     "Les numéros de nœuds n'ont pas de sens d'un graphe à l'autre ; l'information utile (chemin, liens) est captée autrement."),
    ("tar_file", "scenarios", "Archive d'origine", "texte", I, "Traçabilité uniquement."),
    # ---------------- slice / admission ----------------
    ("slice_type", "flows", "Type de slice : eMBB, URLLC ou mMTC", "catégoriel (3)", E,
     "Détermine taille de paquets, débit et file WFQ ; encodé en one-hot (is_eMBB, is_URLLC, is_mMTC)."),
    ("slice_num", "flows", "Numéro de slice dans le scénario", "entier", I, "Identifiant local, pas d'ordre physique."),
    ("delta", "flows", "Facteur de provisionnement de la slice (module d'admission) ; réservation = débit nominal × (0,85 + 0,3·δ)", "réel [0,1]", E,
     "C'est la décision d'allocation : δ<0,5 = sous-provisionnement. Connu avant simulation."),
    ("prov_ratio", "flows (dérivée)", "Réservation / débit réel offert = (0,85+0,3δ)·bw_nominal / rate_bps", "réel", E,
     "Mesure directe du sur/sous-provisionnement par rapport à la charge réelle (eMBB/mMTC toujours < 1)."),
    ("nominal_bw", "flows", "Débit déclaré dans le fichier slices (bits/s)", "réel", X,
     "Sous-estime le débit réel de 25 % pour eMBB/mMTC (JOURNAL D0.3) ; remplacé par rate_bps."),
    # ---------------- traffic generator ----------------
    ("rate_bps", "flows (dérivée)", "Débit offert réel = AvgPktsLambda × taille moyenne de paquet", "réel (bits/s)", E,
     "Charge réellement injectée, calculée depuis les paramètres du générateur ; vérifiée = AvgBw mesuré à 0,2 % près."),
    ("avg_pkts_lambda", "flows", "Paquets générés par seconde (paramètre du générateur)", "réel", E, "Taux d'arrivée λ de la file."),
    ("mean_pkt_size, pkt_size_1/2", "flows", "Taille moyenne / tailles possibles des paquets (bits, loi binomiale)", "réel", E,
     "Fixe le temps de transmission S = taille / capacité."),
    ("eq_lambda, avg_pkt_size_param", "flows", "Paramètres nominaux du générateur", "réel", X,
     "Redondants et incohérents avec les tailles réelles (cf. nominal_bw)."),
    ("time_dist, size_dist, exp_max_factor", "flows", "Lois d'inter-arrivée (Poisson) et de taille (binomiale)", "catégoriel", X,
     "Constantes sur tout le jeu (vérifié en EDA) : aucune information."),
    ("tos", "flows", "Type of Service du flux (= nœud source, vérifié 100 %)", "entier", I,
     "Sert à retrouver la file WFQ du flux à chaque saut ; pas une feature en soi."),
    # ---------------- topology / routing ----------------
    ("path", "flows", "Chemin simulé (reconstruit depuis routing_K.txt)", "liste de nœuds", E,
     "Structure relationnelle : encodée par les arêtes flux→lien du graphe (GNN) ou par des agrégats (modèles plats)."),
    ("path_len", "flows", "Nombre de liens du chemin", "entier", E, "Plus de sauts = plus de files traversées."),
    ("path_matches_slice_file", "flows", "Le chemin simulé = le chemin du fichier slices ?", "booléen", X,
     "Indicateur de diagnostic (EDA) ; son effet est déjà capté par wfq_share_min."),
    ("capacity_bps", "links", "Capacité du lien (bits/s)", "réel", E, "Dénominateur de ρ et du temps de service."),
    ("queue_size_pkts", "flow_links", "Taille de la file de sortie (paquets)", "entier", E, "Borne K de la file M/M/1/K (32 partout ?)."),
    ("wfq_share", "flow_links", "Part WFQ garantie à la file du flux sur le port = poids / Σ poids", "réel [0,1]", E,
     "Ressource effectivement allouée à la slice sur ce lien : c'est l'allocation que le modèle doit évaluer."),
    ("is_default_queue", "flow_links", "Le flux tombe dans la file par défaut du port", "booléen", E, "La slice n'a pas de file dédiée sur ce port."),
    ("levels_qos, sched_policy", "links", "Nombre de files et politique (WFQ) du nœud", "entier / catégoriel", E,
     "Configuration de l'ordonnanceur (politique constante = WFQ)."),
    # ---------------- derived input features ----------------
    ("rho_link_*", "flows (dérivée)", "Charge offerte / capacité des liens du chemin (max, moyenne, somme)", "réel", E,
     "Principe de files d'attente : le délai d'attente croît en 1/(1-ρ)."),
    ("rho_queue_*", "flows (dérivée)", "Charge de la file WFQ / débit garanti (part × capacité)", "réel", E,
     "Vue « débit garanti » de WFQ : explique les flux affamés (part ≈ 0)."),
    ("n_flows_link_*, n_flows_busiest_link", "flows (dérivée)", "Nombre de flux concurrents sur les liens du chemin", "entier", E,
     "Mesure de la compétition pour la ressource partagée."),
    ("mm1_sum, mm1k_*", "flows (dérivée)", "Délais analytiques M/M/1 (ρ écrêté) et M/M/1/K sommés sur le chemin", "réel (s)", E,
     "Connaissance physique injectée ; aussi utilisés seuls comme baseline analytique."),
    ("n_core_nodes, n_flows, n_slices", "scenarios", "Taille du scénario", "entier", E, "Contexte global (taille du réseau, charge totale)."),
    ("max_avg_lambda", "scenarios", "Intensité de trafic globale du simulateur", "réel", X, "Constante sur le jeu (vérifié en EDA)."),
    # ---------------- simulation outputs ----------------
    ("delay", "flows", "Délai moyen par paquet du flux (s)", "réel", C, "Cible principale (régression sur log10)."),
    ("jitter", "flows", "Variance du délai par paquet (s²)", "réel", C, "Cible secondaire."),
    ("loss = pkts_drop / pkts_gen", "flows", "Taux de pertes du flux", "réel [0,1]", C, "Cible secondaire (forte en surcharge)."),
    ("ln_delay, p10…p90", "flows", "Moyenne de ln(délai), percentiles du délai", "réel", X,
     "Autres statistiques de la même distribution que la cible : fuite si utilisées en entrée."),
    ("avg_bw_kbps, pkts_gen, pkts_drop", "flows", "Débit et paquets mesurés pendant la simulation", "réel", X,
     "Sorties de simulation (pkts_drop sert seulement à calculer la cible loss)."),
    ("meas_util, meas_loss, meas_avg_pkt_size", "links", "Utilisation / pertes mesurées par port (linkUsage.txt)", "réel", X,
     "Sorties de simulation : fuite. Servent uniquement à valider ρ (JOURNAL D0.4)."),
    ("global_packets, global_losses, global_delay", "scenarios", "Agrégats mesurés du scénario", "réel", X, "Sorties de simulation : fuite."),
    ("sim_time, sim_conv, sim_mem_kb, sim_wallclock_s", "scenarios", "Méta-données du simulateur", "réel", X,
     "Décrivent la simulation, pas le réseau ; sim_wallclock_s chiffre le coût d'une simulation (motivation)."),
]


def variables_table() -> pd.DataFrame:
    return pd.DataFrame(VARIABLES, columns=["variable", "table", "signification", "type", "rôle potentiel", "justification"])


def save_variables_table() -> pd.DataFrame:
    df = variables_table()
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "variables.csv", index=False, encoding="utf-8")
    md = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    md += ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False)]
    (ROOT / "reports" / "variables.md").write_text("# Dictionnaire des variables (L1)\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    return df
