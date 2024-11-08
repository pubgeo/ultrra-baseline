<h3>Competition Overview</h3>
<p>The ULTRRA challenge evaluates current and novel state of the art view synthesis methods for posed and unposed cameras. Challenge datasets emphasize real-world considerations, such as image sparsity, variety of camera models, and unconstrained acquisition in real-world environments.</p>
<h4>Schedule:</h4>
<ul>
	<li>Development dataset release: 11/8/2024</li>
	<li>Challenge dataset release: 1/10/2025</li>
	<li>Submission period: 11/1/2024 - 1/31/2025</li>
	<li>Invited presentations for selected participants: 2/14/2025</li>
</ul>
<p>To get started, please register for the competition and download the development data package from <a href="https://ieee-dataport.org/competitions/ultrra-challenge-2025">IEEE DataPort</a>.</p>
<h4>Tasks:</h4>
	<h5>Camera calibration</h5>
	<ul>
	<li>Inputs: unposed images</li>
	<li>Ouputs: relative camera extrinsic parameters</li>
	<li>Evaluation: camera geolocation error</li>
</ul>
	<h5>View synthesis</h5>
	<ul>
	<li>Inputs: images with camera locations, camera metadata for requested novel image renders</li>
	<li>Outputs: rendered images</li>
	<li>Evaluation: DreamSim image similarity metric</li>
</ul>
<h4>Challenges posed for each task, increasing in complexity:</h4>
<ul>
	<li>Image density: a limited number of input images from a single ground-level camera</li>
	<li>Multiple camera models: images from multiple unique ground-level cameras</li>
	<li>Varying altitudes: images from ground, security, and airborne cameras</li>
	<li>Reconstructed area: images from varying altitudes, covering a larger area</li>
</ul>
<p>Example datasets are provided for each task and challenge to support algorithm development. An example baseline solution is provided based on COLMAP and NerfStudio, and a baseline submission is provided to clarify the expected submission format.</p>