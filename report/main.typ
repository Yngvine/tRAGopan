
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
      affiliation: [University of Navarre: NLP Subject]
    ),
  ),
)


#abstract[
  Geographic Information Systems (GIS) provide rich cartographic and administrative data but often require specialized expertise to operate. We present IDENA-SVP (Search and Visualization Platform), an agent designed for geographic information retrieval focused on the Navarre region. Our system integrates spatial data from the IDENA (Infrastructure for Spatial Data in Navarre) Web Feature Service (>100 layer endpoints) equipped with Large Language Models to enable natural language interaction. Rather than comparing against a weak algorithmic baseline, we establish a rigorous human baseline: the manual GIS workflow required by a professional user to retrieve the same geographic insights. Using strict data isolation procedures, we evaluate the comparative retrieval latency, accuracy, and operational friction. Finally, through a detailed analysis of our Top 20 worst system failures, we characterize boundary ambiguities and complex multi-step reasoning limits in contemporary spatial LLMs.
]


= Introduction

Geographic Information Systems (GIS) have traditionally required high technical fluency, bounding spatial data access to expert domains. However, advances in Large Language Models (LLMs) equipped with tool-use affordances open the possibility of democratizing spatial information through conversational interfaces. This capstone presents IDENA-SVP, an orchestrated agent bridging natural language and spatial data via the IDENA Web Feature Service (WFS) @idena_wfs for the Navarre region.

Navarre maintains extensive open cartographic data—municipal boundaries, topographical landmarks, and utility networks. The traditional method for answering geographic queries (e.g., “What municipalities border Estella?”) requires navigating complex portal UIs, pulling raw GeoJSONs, or manually clipping spatial features in QGIS. IDENA-SVP replaces this workflow by deploying an LLM-driven autonomous agent capable of dynamically forming WFS queries, resolving ambiguous user prompts, and executing spatial joints. 

The primary contribution of this work is an end-to-end evaluation of conversational GIS that emphasizes rigorous benchmarking. Specifically, this paper outlines: (1) seamless integration with >100 IDENA WFS documents, (2) comparative evaluation against human-driven manual GIS baselines, and (3) an explicit root-cause breakdown of the system’s top 20 worst failures to guide future spatial-LLM design.


= Related Work

Tool-augmented generation and autonomous agents significantly reduce LLM hallucinations by grounding generation in verifiable external API executions @schick2023toolformerlanguagemodelsteach. Extending these agents to spatial domains introduces complex routing challenges, as spatial queries require both semantic matching and geometric computations (e.g., bounding boxes or spatial intersections). 

While traditional semantic search relies on static vector embeddings, geographic retrieval requires dynamic orchestration of explicit parameters. In our case, the LLM actively parses intent into structured WFS filters. However, most baseline evaluations in NLP rely on rudimentary static models (e.g., TF-IDF). In applied agentic geographic retrieval, an algorithmic baseline often misses the primary friction point the system solves: human usability. Therefore, aligned with recent HCI-focused evaluations, we depart from non-Deep Learning predictive baselines and instead use the manual human operational ease of use as our primary benchmark.


= Methodology

== Data Infrastructure and Integration

Rather than relying on synthetic or toy datasets, we operate directly on live, authoritative geographic data. Our system interfaces directly with the IDENA Web Feature Service endpoint over the Navarre region. 

The underlying data encompasses over >100 distinct spatial documents (layers / `typeNames`), including:
- `IDENA:DIADMI_Pol_Municipio` (Municipal polygons)
- `IDENA:DIADMI_Pol_Concejo` (Council boundaries)
- `IDENA:TOPONI_Txt_Toponimos` (Toponymic points)
- Infrastructure, Utility, and Agricultural layers.

== Agentic Orchestration Architecture

Because agentic workflows over GIS are highly multimodal, combining semantic filtering with explicit spatial geometry, we constructed a specialized `AgentGeoService` powered by a LangGraph @langgraph2024 state machine. This architecture allows the LLM to orchestrate multi-step spatial reasoning deterministically without monolithic code generation.

Execution initiates via a `parse_prompt` node, which categorizes the user's natural language intent (e.g., `municipality_boundary`, `filtered_layer`, `toponymy_in_municipality`) and determines whether the query scope is localized or Navarra-wide. Depending on the parsed configuration, the system state routes through highly specialized processing nodes:
- *Entity Resolution* (`resolve_municipality`): Extracts text references to Navarrese municipalities and statically resolves their exact polygonal geometries for downstream operational bounding.
- *Semantic Layer Matching* (`load_capabilities` $->$ `select_layer`): Dynamically fetches the authoritative WFS catalog and scores the user's query against 100+ layer endpoints to select the target feature. If the request is ambiguous, it populates a `layer_candidates` list for a human-in-the-loop clarification prompt.
- *Spatial Extraction* (`fetch_features` $->$ `spatial_filter`): Executes paginated WFS queries using extracted bounding boxes. Since bounding boxes are imperfect rectangles, the backend systematically applies a spatial intersection filter to cleanly exclude any features falling outside the true irregular municipal boundaries.

Finally, all execution trajectories converge on a terminal `finalize` node, which packages the underlying `filtered_geojson` payload into web-map actions, updates the internal dialogue context, and synthesizes the natural-language response.

#figure(
  image("resources/agent_graph.png", width:100%),
  caption:[IDENA-SVP's Agentic Orchestration Architecture, showcasing the LangGraph state machine with specialized nodes for parsing, entity resolution, layer selection, spatial extraction, and finalization. The architecture enables deterministic multi-step reasoning over complex spatial queries while maintaining a clear separation of concerns across processing stages.],
  placement: auto,
  scope: "parent"
) <fig:agent_architecture>

== Evaluation Baseline: Manual GIS Workflow

Standard ML benchmarks often require algorithmic predictive baselines (e.g., TF-IDF). However, because IDENA-SVP functions as a tool-orchestrated retrieval engine, standard classification baselines fail to capture the workflow improvements it offers. Therefore, our primary benchmark is the *knowledge volume and operational friction a user endures to manually retrieve the same information*.

*Knowledge Stack Required (Manual Baseline)*:
- *Geospatial Data Models & CRS*: Understanding vector topologies and Coordinate Reference System transformations (e.g., native EPSG:25830 to web EPSG:4326).
- *OGC Web Services*: Differentiating raw vector API extraction (WFS) from rendered images (WMS).
- *API & Query Dynamics*: Formatting complex WFS parameters (`cql_filter`, `bbox`) and managing pagination limits (2000 features per request).
- *Tooling*: Proficiency in Desktop GIS software (e.g., QGIS) or REST API clients.

*General Procedure Loop*:
1. *Discovery & Connection*: Manually navigate the IDENA catalog, connect to the WFS endpoint, and pull the layer.
2. *Filtering & Extraction*: Isolate the bounding geographic entity, extract its bounding box (minX, minY, maxX, maxY), and execute a paginated request for target points inside that box.
3. *Geoprocessing & Synthesis*: Load data into QGIS, spatially intersect/clip the bounding-box perimeter against the exact irregular polygon, and manually examine the attribute table to read out the answer.

*Experimental System (IDENA-SVP)*:
1. User types intent into conversational UI ("Show me hydrants in Olite").
2. The LLM intelligently orchestrates the entire underlying WFS extraction and geoprocessing pipeline, instantly mapping the synthesized text array.

We quantify the baseline using "Time to Insight" (Query latency) and Operational Steps, comparing expert manual GIS operators to novice users leveraging IDENA-SVP.


= Experiments

== Setup
We benchmarked IDENA-SVP using a suite of 100 geographically diverse queries. These were subdivided into:
- *Simple Entity Location* ("Load the boundary of Tafalla")
- *Constrained Feature Retrieval* ("Find toponyms in Estella-Lizarra")
- *Fuzzy Capability Search* ("Show me available layers about agriculture")
- *Spatial Intersection* ("Show me utility network pipes inside Burlada")


The system is deployed heavily leveraging the EIM-Laboratory AI server, running Ollama @ollama2024 on an NVIDIA RTX PRO 6000 Blackwell GPU (see @fig:architecture). 
#figure(
  image("resources/big_picture_graph.png", width:100%),
  caption:[IDENA-SVP's architecture, showcasing the integration of the LLM with the IDENA WFS and the user interface. The LLM serves as the central orchestrator, interpreting user queries and translating them into structured WFS calls to retrieve and visualize spatial data.],
  placement: auto,
  scope: "parent"
) <fig:architecture>
We expose a connection to that Ollama service via Tailscale. The frontend GUI connects to the project backend, which routes intent through the remote GPU while simultaneously executing WFS calls to IDENA. As our "remote brain," the system initially utilized the Qwen3.5-35B model @qwen2024. However, because this model severely constrained the Virtual Machine's storage capacity and introduced high inference latency, we ultimately migrated to Llama3-7B @llama3_2024, which provided a significant improvement in response speed while maintaining the necessary reasoning capabilities for tool orchestration.


== Qualitative Analysis

=== Chain of Thought Transparency

A critical hurdle in adopting LLM agents for professional GIS workflows is the "black box" nature of autonomous tool execution. To establish user trust and facilitate debugging, IDENA-SVP surfaces its internal reasoning via a real-time Chain of Thought (CoT) trace. Rather than a monolithic loading state, the system presents sequential logic transitions corresponding to the active LangGraph nodes.

For example, when a user queries *"Muestrame la capa de acometidas de Pamplona"* (Show me the utility connections layer for Pamplona), the explicit trace exposes the deterministic pipeline:
1. *Intent Parsing*: The system infers the correct operational mode (`FILTERED_LAYER`).
2. *Entity Resolution*: It extracts the target region and statically resolves it to the official spatial boundary ("PAMPLONA / IRUÑA").
3. *Semantic Layer Matching*: The agent queries the WFS catalog with the concept, surfacing candidate layers before selecting the optimal match (`IDENA:REDSAN_LIN_ACOMETIDA`).
4. *Spatial Extraction & Rendering*: The system applies the spatial filter intersecting the municipal boundary, explicitly reporting the resulting feature count (e.g., 238 features) prior to the final web map render.
#figure(
  image("resources/transparent_cot.png", width:70%),
  caption:[IDENA-SVP's Chain of Thought (CoT) trace, showcasing the step-by-step reasoning process as the agent parses user intent, resolves entities, matches layers, and executes spatial extraction. The CoT trace provides real-time visibility into the agent's internal logic, enabling users to understand and verify each stage of the geographic information retrieval process.],
) <fig:cot_trace>

By making the agent's internal trajectory visible, users can instantly pinpoint execution failures—whether the LLM hallucinated a region, selected an incorrect WFS layer, or dropped features during the spatial intersection. This transparency drastically reduces the friction of verifying AI-generated geographic insights.

=== Examples

#figure(
  grid(
    columns:2,
    gutter: 10pt,
    image("resources/no_filter_example.jpeg", width: 98.5%),
    image("resources/filter_example.jpeg"),
  ),
  caption:[Example of a working query in IDENA-SVP, demonstrating the system's ability to correctly interpret a user's request and execute the appropriate WFS queries to retrieve and visualize spatial data. The left image shows a query without spatial filtering, resulting in a broader result set, while the right image illustrates a query with spatial filtering, yielding only the results within the specified municipality.], 
  placement: auto,
  scope: "parent"
) <fig:query_comparison>

On @fig:query_comparison we can see how the contrast between these two execution states highlights the agent's dynamic spatial-awareness capabilities. In the unrestricted query (top: *"muestrame la capa de acometidas"*), the agent identifies the target semantic layer but correctly recognizes the absence of a localized geographical entity. It defaults to a region-wide WFS fetch, actively warning the user in the UI that no spatial constraints are applied and displaying a raw dump of 500 data points scattered across the province. 

However, when the user appends an explicit territorial constraint (bottom: *"muestrame la capa de acometidas en pamplona"*), the LangGraph orchestration triggers the `resolve_municipality` and `spatial_filter` nodes. This contextual injection produces a highly precise outcome: the system visually anchors the canvas with the irregular municipal bounding polygon of "Pamplona / Iruña" and strictly returns the 366 data points intersecting that exact geometry. This juxtaposition showcases how seamlessly the agent handles the transition from global to localized spatial extraction without requiring the user to manually clip layers.

=== Error Analysis

To understand systemic limitations beyond raw accuracy metrics, we conducted an explicit root-cause breakdown of IDENA-SVP’s failure modes, comparing the top 20 failure cases across both evaluated models (Qwen3.5-35B and Llama3-7B).

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: center + horizon,
    [*Mode*], [*Qwen3.5-35B*], [*Llama3-7B*],
    [1], [2], [4],
    [2], [7], [8],
    [3], [2], [8]
  ),
  caption: [Categorization of the top 20 failure cases per model. Mode (1): Spurious Spatial Cross-talk; Mode (2): Pagination & Density Limits; Mode (3): Model Hallucination. Despite Llama3-7B providing necessary operational speed improvements, both models exhibited similar structural limitations regarding topological reasoning and dense spatial data.]
) <tab:error_comparison>

The failures categorized primarily into three root causes, as detailed in @tab:error_comparison:

1. *Spurious Spatial Cross-talk*: Complex queries involving negation or exclusion ("Show parcels EXCEPT in Zone A") confused the WFS query builder. The LLM lacked native understanding of topological negation and defaulted to an unconstrained query. (e.g., "Show me all toponyms in Navarre EXCEPT those in Pamplona" -> LLM fails to apply the spatial exclusion filter, resulting in a complete dump of all toponyms or alternatively the toponyms present in Pamplona).
2. *Pagination & Density Limits*: For highly dense layers (like wide-area toponymy), the WFS limit of 2000 features truncated results before the LLM could summarize the correct region. The underlying map correctly plotted partial data, but the LLM confidently hallucinated completeness. (e.g., "Show me all toponyms in the province of Navarre" -> LLM reports 2000 features but fails to clarify that this is a truncated subset of the true total).
3. *Model Hallucination*: The most common failure mode was the LLM breaking the deterministic orchestration by generating incorrect information or tool parameters (e.g., wrong municipality name, incorrect layer selection, or malformed WFS filters). (e.g., "Show me toponyms in Villava" -> LLM determines it can't answer the query because "Municipality 'Villava' belongs to the province of Bizkaia").



= Conclusion and Future Work

IDENA-SVP successfully demonstrates an alternative to traditional, high-friction GIS interfaces. By bridging LLM routing with IDENA WFS infrastructure, we out-performed manual GIS baseline workflows in both execution time and accessibility for non-expert users. Our failure analysis reveals that while simple geometric intersection is highly reliable via Agentic LLMs, resolving topological negation and parsing overlapping feature bounds requires teh usage of either a more capable model or better integration strategies.

As a primary avenue for future work, we plan to revisit the project's original inception: *tRAGopan* (Topographic Retrieval-Augmented Generation). The initial vision sought to integrate the current tool-centric platform with a dedicated vector database containing dense semantic embeddings of the topographic surface and regions. This approach would equip the agent with rich, continuous spatial-semantic context to deeply understand the geographic terrain it operates on. While this vector-embedding component was ultimately descoped due to development time constraints and unfeasible storage requirements on the project's Virtual Machine, supplementing IDENA-SVP's current structured WFS queries with such an embedding footprint represents the natural next step toward fully holistic geographic reasoning.


#bibliography("references.bib")
