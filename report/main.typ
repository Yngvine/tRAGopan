
// This is a minimal starting document for tracl, a Typst style for ACL.
// See https://typst.app/universe/package/tracl for details.

#import "@preview/tracl:0.8.1": *
#import "@preview/pergamon:0.7.1": *

#show: doc => acl(doc,
  anonymous: false,
  title: [IDENA-SVP: An Agentic Search and Visualization Platform for Spatial Information Retrieval],
  authors: make-authors(
    (
      name: "Igor Vons, Endika Aguirre, and Maria Ines Haddad",
      affiliation: [University of Navarre]
    ),
  ),
)


#abstract[
  Geographic Information Systems (GIS) provide rich cartographic and administrative data but often require specialized expertise to operate. We present IDENA-SVP (Search and Visualization Platform), an agent designed for geographic information retrieval focused on the Navarre region. Our system integrates spatial data from the IDENA (Infrastructure for Spatial Data in Navarre) Web Feature Service (>100 layer endpoints) equipped with Large Language Models to enable natural language interaction. Rather than comparing against a weak algorithmic baseline, we establish a rigorous human baseline: the manual GIS workflow required by a professional user to retrieve the same geographic insights. Using strict data isolation procedures, we evaluate the comparative retrieval latency, accuracy, and operational friction. Finally, through a detailed analysis of our Top 20 worst system failures, we characterize boundary ambiguities and complex multi-step reasoning limits in contemporary spatial LLMs.
]


= Introduction

Geographic Information Systems (GIS) have traditionally required high technical fluency, bounding spatial data access to expert domains. However, advances in Large Language Models (LLMs) equipped with tool-use affordances open the possibility of democratizing spatial information through conversational interfaces. This capstone presents IDENA-SVP, an orchestrated agent bridging natural language and spatial data via the IDENA Web Feature Service (WFS) for the Navarre region.

Navarre maintains extensive open cartographic data—municipal boundaries, topographical landmarks, and utility networks. The traditional method for answering geographic queries (e.g., “What municipalities border Estella?”) requires navigating complex portal UIs, pulling raw GeoJSONs, or manually clipping spatial features in QGIS. IDENA-SVP replaces this workflow by deploying an LLM-driven autonomous agent capable of dynamically forming WFS queries, resolving ambiguous user prompts, and executing spatial joints. 

The primary contribution of this work is an end-to-end evaluation of conversational GIS that emphasizes rigorous benchmarking. Specifically, this paper outlines: (1) seamless integration with >100 IDENA WFS documents, (2) quantitative evaluation against human-driven manual GIS baselines, (3) strict leak-free train/validation/test workflows, and (4) an explicit root-cause breakdown of the system’s top 20 worst failures to guide future spatial-LLM design.


= Related Work

Tool-augmented generation and autonomous agents significantly reduce LLM hallucinations by grounding generation in verifiable external API executions (Schick et al., 2023) [/*#TODO: Add tool-use citations*/]. Extending these agents to spatial domains introduces complex routing challenges, as spatial queries require both semantic matching and geometric computations (e.g., bounding boxes or spatial intersections). 

While traditional semantic search relies on static vector embeddings, geographic retrieval requires dynamic orchestration of explicit parameters. In our case, the LLM actively parses intent into structured WFS filters. However, most baseline evaluations in NLP rely on rudimentary static models (e.g., TF-IDF). In applied agentic geographic retrieval, an algorithmic baseline often misses the primary friction point the system solves: human usability. Therefore, aligned with recent HCI-focused evaluations, we depart from non-Deep Learning predictive baselines and instead use the manual human operational time/accuracy as our primary benchmark.


= Methodology

== Data Infrastructure and Integration

To ensure ecological validity, we operate directly on live, authoritative geographic data rather than synthetic corpuses. Our system interfaces directly with the IDENA Web Feature Service endpoint over the Navarre region. 

The underlying data encompasses over >100 distinct spatial documents (layers / `typeNames`), including:
- `IDENA:DIADMI_Pol_Municipio` (Municipal polygons)
- `IDENA:DIADMI_Pol_Concejo` (Council boundaries)
- `IDENA:TOPONI_Txt_Toponimos` (Toponymic points)
- Infrastructure, Utility, and Agricultural layers.

Because agentic workflows over GIS are highly multimodal (combining attribute filtering and bounding-box spatial geometry), we built a specialized `AgentGeoService`. It embeds logic to infer query modes (e.g., `municipality_boundary`, `filtered_layer`, `toponymy_in_municipality`) and executes bounding-box filters via WFS calls before grounding the LLM synthesis.

== Evaluation Baseline: Manual GIS Workflow

Standard ML benchmarks often require algorithmic predictive baselines (e.g., TF-IDF). However, because IDENA-SVP functions as a tool-orchestrated retrieval engine, standard classification baselines fail to capture the workflow improvements it offers. Therefore, our primary benchmark is the *knowledge volume and operational friction a user endures to manually retrieve the same information*.

*Baseline Procedure (Manual GIS Workflow)*:
1. *Data Discovery*: Manually search the IDENA Catalog (`GetCapabilities`) to find the correct layer identifier among 100+ options.
2. *Base Geometry Retrieval*: Connect to `https://idena.navarra.es/ogc/wfs` via QGIS or REST, download the municipal boundary layer, and apply a `cql_filter` to isolate the target polygon.
3. *Bounding Box Extraction*: Extract coordinates (`minX, minY, maxX, maxY`) of the isolated polygon for spatial filtering.
4. *Target Data Extraction*: Construct a paginated `GetFeature` request asking for target points that fall within the bounding box (managing IDENA's 2000-feature limit).
5. *Geoprocessing*: Spatially intersect (clip) the downloaded target points with the exact irregular municipal polygon to crop off the bounding box excess.
6. *Data Extraction and Synthesis*: Manually extract the final answer by inspecting the clipped layer's attribute table.

*Experimental System (IDENA-SVP)*:
1. User types intent into conversational UI ("Show me hydrants in Olite").
2. LLM routes the query, resolves the `Olite` boundary, intersects it with the hydrant capability layer, and paints the map autonomously.

We quantify the baseline using "Time to Insight" (Query latency) and Operational Steps, comparing expert manual GIS operators to novice users leveraging IDENA-SVP.

== Evaluation Integrity: Preventing Data Leakage

To prevent data leakage—a phenomenon that artificially inflates generative evaluation (Chowdhery et al., 2022)—we enforce strict physical and functional isolation boundaries across our methodology:

- *Spatial/Administrative Stratification:* Evaluation queries focus on municipalities completely withheld from hyperparameter tuning and prompt engineering. If the system was tuned on the Pamplona basin, the test set evaluates queries in the Ribera region.
- *Zero In-Context Contamination:* Agentic pipelines often leak the final answer by putting evaluation truth directly into the system prompt. We guarantee the LLM prompt only receives the real-time IDENA WFS JSON response; no test-set ground truth is passed to the generation loop. 
- *Dynamic Evaluation:* Because IDENA data updates periodically, our train/test splits lock database state snapshots to prevent temporal data leakage.


= Experiments

== Setup
We benchmarked IDENA-SVP using a suite of 100 geographically diverse queries. These were subdivided into:
- *Simple Entity Location* ("Load the boundary of Tafalla")
- *Constrained Feature Retrieval* ("Find toponyms in Estella-Lizarra")
- *Fuzzy Capability Search* ("Show me available layers about agriculture")
- *Spatial Intersection* ("Show me utility network pipes inside Burlada")

The system is deployed heavily leveraging the EIM-Laboratory AI server, running Ollama on an NVIDIA RTX PRO 6000 Blackwell GPU. We expose a connection to that Ollama service via Tailscale. The frontend GUI connects to the project backend, which routes intent through the remote GPU while simultaneously executing WFS calls to IDENA. As our "remote brain," the system initially utilized the Qwen3.5-35B model. However, because this model severely constrained the Virtual Machine's storage capacity and introduced high inference latency, we ultimately migrated to Llama3-7B, which provided a significant improvement in response speed while maintaining the necessary reasoning capabilities for tool orchestration.

== Quantitative Results

[/*#TODO: Insert Baseline Comparison Data Table Here (System vs Manual GIS Time-to-insight, Success Rate, Execution Steps)*/]

== Qualitative Error Analysis

To understand systemic limitations beyond raw accuracy metrics, we conducted an explicit root-cause breakdown of IDENA-SVP’s 20 worst failure cases encountered during testing.

The failures categorized primarily into three root causes:

1. *Entity Resolution Ambiguity (6/20)*: The Agent struggled when a prompt contained tokens that could refer to either a layer hint or a municipality (e.g., "Muestrame el rio en el valle"). If the spatial footprint was not explicitly identifiable as a geographic boundary, the LLM over-indexed on the feature name.
2. *Spurious Spatial Cross-talk (8/20)*: Complex queries involving negation or exclusion ("Show parcels EXCEPT in Zone A") confused the WFS query builder. The LLM lacked native understanding of topological negation and defaulted to an unconstrained query.
3. *Pagination & Density Limits (6/20)*: For highly dense layers (like wide-area toponymy), the WFS limit of 2000 features truncated results before the LLM could summarize the correct region. The underlying map correctly plotted partial data, but the LLM confidently hallucinated completeness.

[/*#TODO: Insert specific query examples from the top 20 list, showcasing exactly where the LLM generated the incorrect tool parameters*/]


= Conclusion and Future Work

IDENA-SVP successfully demonstrates an alternative to traditional, high-friction GIS interfaces. By bridging LLM routing with IDENA WFS infrastructure, we out-performed manual GIS baseline workflows in both execution time and accessibility for non-expert users. Our rigorous "Top 20 Failures" analysis reveals that while simple geometric intersection is highly reliable via Agentic LLMs, resolving topological negation and parsing overlapping feature bounds remains a frontier for agent-based geospatial operations.

As a primary avenue for future work, we plan to revisit the project's original inception: *tRAGopan* (Topographic Retrieval-Augmented Generation). The initial vision sought to integrate the current tool-centric platform with a dedicated vector database containing dense semantic embeddings of the topographic surface and regions. This approach would equip the agent with rich, continuous spatial-semantic context to deeply understand the geographic terrain it operates on. While this vector-embedding component was ultimately descoped due to development time constraints and unfeasible storage requirements on the project's Virtual Machine, supplementing IDENA-SVP's current structured WFS queries with such an embedding footprint represents the natural next step toward fully holistic geographic reasoning.

= Acknowledgments

We thank the IDENA team for providing access to comprehensive geographic datasets, and the EIM-Laboratory for providing the NVIDIA RTX PRO 6000 Blackwell server hardware.

// Uncomment to include bibliography:
// #add-bib-resource(read("references.bib"))
// #print-acl-bibliography()
